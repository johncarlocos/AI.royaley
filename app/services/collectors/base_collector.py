"""
ROYALEY - Base Collector Framework
Phase 1: Data Collection Services

Abstract base class for all data collectors with rate limiting and retry logic.
Every API response is automatically archived to HDD (16TB) via RawDataArchiver.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Generic, List, Optional, TypeVar

import httpx

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class RateLimiter:
    """Token bucket rate limiter with sliding window."""
    
    max_requests: int
    window_seconds: int
    requests: List[float] = field(default_factory=list)
    
    def can_request(self) -> bool:
        """Check if a request can be made."""
        self._cleanup()
        return len(self.requests) < self.max_requests
    
    def add_request(self) -> None:
        """Record a request."""
        self.requests.append(time.time())
    
    def wait_time(self) -> float:
        """Get time to wait before next request."""
        self._cleanup()
        if len(self.requests) < self.max_requests:
            return 0.0
        oldest = min(self.requests)
        return max(0.0, oldest + self.window_seconds - time.time())
    
    def _cleanup(self) -> None:
        """Remove expired requests from window."""
        cutoff = time.time() - self.window_seconds
        self.requests = [r for r in self.requests if r > cutoff]


@dataclass
class RetryStrategy:
    """Exponential backoff retry strategy."""
    
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    multiplier: float = 2.0
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt."""
        delay = self.base_delay * (self.multiplier ** attempt)
        return min(delay, self.max_delay)


@dataclass
class CollectorResult(Generic[T]):
    """Result wrapper for collector operations."""
    
    success: bool
    data: Optional[T] = None
    error: Optional[str] = None
    records_count: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseCollector(ABC):
    """
    Abstract base class for data collectors.
    
    Provides:
    - HTTP client with connection pooling
    - Rate limiting
    - Retry logic with exponential backoff
    - Caching integration
    """
    
    def __init__(
        self,
        name: str,
        base_url: str,
        rate_limit: int = 100,
        rate_window: int = 60,
        timeout: float = 30.0,
        max_retries: int = 3,
    ):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.rate_limiter = RateLimiter(rate_limit, rate_window)
        self.retry_strategy = RetryStrategy(max_retries=max_retries)
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
    
    async def get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
            )
        return self._client
    
    async def close(self) -> None:
        """Close HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers. Override in subclasses for auth."""
        return {"Accept": "application/json"}
    
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Make HTTP request with rate limiting and retry logic.
        Every successful response is automatically archived to HDD.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            params: Query parameters
            json_data: JSON body data
            
        Returns:
            Parsed JSON response
            
        Raises:
            Exception: If all retries fail
        """
        # Rate limiting
        wait_time = self.rate_limiter.wait_time()
        if wait_time > 0:
            logger.debug(f"[{self.name}] Rate limit: waiting {wait_time:.2f}s")
            await asyncio.sleep(wait_time)
        
        client = await self.get_client()
        last_error = None
        
        for attempt in range(self.retry_strategy.max_retries + 1):
            try:
                self.rate_limiter.add_request()
                
                # Log the request
                full_url = f"{self.base_url}{endpoint}"
                logger.info(f"[{self.name}] 🌐 Making {method} request to: {full_url}")
                print(f"[{self.name}] 🌐 Making {method} request to: {full_url}")
                if params:
                    logger.debug(f"[{self.name}] Request params: {params}")
                    print(f"[{self.name}] Request params: {params}")
                
                response = await client.request(
                    method=method,
                    url=endpoint,
                    params=params,
                    json=json_data,
                    headers=self._get_headers(),
                )
                
                logger.info(f"[{self.name}] 📥 Response status: {response.status_code}")
                print(f"[{self.name}] 📥 Response status: {response.status_code}")
                
                # Handle rate limit response
                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", 60))
                    logger.warning(f"[{self.name}] Rate limited, waiting {retry_after}s")
                    await asyncio.sleep(retry_after)
                    continue
                
                response.raise_for_status()
                response_data = response.json()
                logger.info(f"[{self.name}] ✅ Successfully fetched data: {len(response_data) if isinstance(response_data, list) else 'object'} items")
                print(f"[{self.name}] ✅ HTTP {response.status_code} - Received data")
                
                # ============================================================
                # AUTO-ARCHIVE: Save raw response to HDD (non-blocking)
                # ============================================================
                try:
                    from app.services.data.raw_data_archiver import get_archiver
                    archiver = get_archiver()
                    if archiver.enabled:
                        # Detect sport_code from params or endpoint
                        sport_code = self._detect_sport_code(endpoint, params)
                        # Detect data_type from endpoint
                        data_type = self._detect_data_type(endpoint)
                        
                        # Archive in background (fire-and-forget, don't slow down collection)
                        asyncio.create_task(
                            archiver.archive_api_response(
                                source=self.name,
                                sport_code=sport_code,
                                data=response_data,
                                data_type=data_type,
                                endpoint=endpoint,
                                params=params,
                                response_status=response.status_code,
                            )
                        )
                except Exception as archive_err:
                    # Never let archive failure break data collection
                    logger.debug(f"[{self.name}] Archive skipped: {archive_err}")
                
                return response_data
                
            except httpx.HTTPStatusError as e:
                last_error = e
                logger.error(f"[{self.name}] HTTP {e.response.status_code}: {e}")
                
                # Don't retry client errors (4xx except 429)
                if 400 <= e.response.status_code < 500 and e.response.status_code != 429:
                    raise
                    
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                last_error = e
                logger.warning(f"[{self.name}] Connection error: {e}")
                
            except Exception as e:
                last_error = e
                logger.error(f"[{self.name}] Unexpected error: {e}")
            
            # Retry with backoff
            if attempt < self.retry_strategy.max_retries:
                delay = self.retry_strategy.get_delay(attempt)
                logger.info(f"[{self.name}] Retry {attempt + 1}/{self.retry_strategy.max_retries} in {delay:.1f}s")
                await asyncio.sleep(delay)
        
        raise Exception(f"[{self.name}] All retries failed: {last_error}")
    
    async def get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Make GET request."""
        return await self._make_request("GET", endpoint, params=params)
    
    async def post(
        self,
        endpoint: str,
        json_data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Make POST request."""
        return await self._make_request("POST", endpoint, params=params, json_data=json_data)
    
    @abstractmethod
    async def collect(self, **kwargs) -> CollectorResult:
        """
        Collect data from source.
        
        Must be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    async def validate(self, data: Any) -> bool:
        """
        Validate collected data.
        
        Must be implemented by subclasses.
        """
        pass
    
    def _detect_sport_code(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Auto-detect sport code from the API endpoint or params.
        Used by the auto-archive system to organize files by sport.
        """
        # Check params first
        if params:
            for key in ["sport", "sport_code", "league", "sportKey", "sport_key"]:
                if key in params and params[key]:
                    return str(params[key]).upper()
        
        # Check endpoint path for known sport identifiers
        endpoint_lower = endpoint.lower()
        sport_keywords = {
            "nfl": "NFL", "football/nfl": "NFL",
            "nba": "NBA", "basketball/nba": "NBA",
            "mlb": "MLB", "baseball/mlb": "MLB",
            "nhl": "NHL", "hockey/nhl": "NHL",
            "ncaaf": "NCAAF", "college-football": "NCAAF",
            "ncaab": "NCAAB", "mens-college-basketball": "NCAAB",
            "wnba": "WNBA",
            "cfl": "CFL",
            "atp": "ATP", "wta": "WTA", "tennis": "TENNIS",
        }
        
        for keyword, code in sport_keywords.items():
            if keyword in endpoint_lower:
                return code
        
        return None
    
    def _detect_data_type(self, endpoint: str) -> str:
        """
        Auto-detect data type from the API endpoint.
        Used by the auto-archive system for file naming.
        """
        endpoint_lower = endpoint.lower()
        
        type_keywords = [
            "scoreboard", "scores", "schedule", "standings",
            "teams", "roster", "players", "injuries",
            "odds", "lines", "spreads", "totals",
            "stats", "boxscore", "play-by-play", "pbp",
            "rankings", "ratings", "weather", "forecast",
            "news", "transactions", "trades",
        ]
        
        for keyword in type_keywords:
            if keyword in endpoint_lower:
                return keyword
        
        # Extract last meaningful path segment
        parts = [p for p in endpoint.strip("/").split("/") if p]
        if parts:
            return parts[-1][:50]
        
        return "response"
    
    async def cache_result(
        self,
        key: str,
        data: Any,
        ttl: int = 300,
    ) -> None:
        """Cache collection result."""
        from app.core.cache import cache
        await cache.set(f"collector:{self.name}:{key}", data, ttl)
    
    async def get_cached(self, key: str) -> Optional[Any]:
        """Get cached result."""
        from app.core.cache import cache
        return await cache.get(f"collector:{self.name}:{key}")


class CollectorManager:
    """
    Manager for multiple collectors.
    
    Provides centralized access and coordination.
    """
    
    def __init__(self):
        self._collectors: Dict[str, BaseCollector] = {}
    
    def register(self, collector: BaseCollector) -> None:
        """Register a collector."""
        self._collectors[collector.name] = collector
        logger.info(f"Registered collector: {collector.name}")
    
    def get(self, name: str) -> Optional[BaseCollector]:
        """Get collector by name."""
        return self._collectors.get(name)
    
    @property
    def collectors(self) -> Dict[str, BaseCollector]:
        """Get all registered collectors."""
        return self._collectors
    
    async def initialize_all(self) -> None:
        """Initialize all collectors."""
        for name, collector in self._collectors.items():
            try:
                await collector.get_client()
                logger.info(f"Initialized collector: {name}")
            except Exception as e:
                logger.error(f"Failed to initialize {name}: {e}")
    
    async def close_all(self) -> None:
        """Close all collectors."""
        for name, collector in self._collectors.items():
            try:
                await collector.close()
                logger.info(f"Closed collector: {name}")
            except Exception as e:
                logger.error(f"Error closing {name}: {e}")
    
    async def collect_all(
        self,
        sport_code: Optional[str] = None,
    ) -> Dict[str, CollectorResult]:
        """
        Run all collectors.
        
        Args:
            sport_code: Optional sport to filter collection
            
        Returns:
            Dict mapping collector names to results
        """
        results = {}
        
        for name, collector in self._collectors.items():
            try:
                result = await collector.collect(sport_code=sport_code)
                results[name] = result
            except Exception as e:
                logger.error(f"Collector {name} failed: {e}")
                results[name] = CollectorResult(
                    success=False,
                    error=str(e),
                )
        
        return results


# Global collector manager instance
collector_manager = CollectorManager()