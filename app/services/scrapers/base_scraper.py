"""
Base Web Scraper Module
=======================
Base class for all web scrapers with common functionality including:
- Rate limiting
- Retry logic with exponential backoff
- User agent rotation
- Proxy support
- Session management
- Data validation
- Error handling
"""

import asyncio
import random
import logging
import hashlib
import json
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import aiohttp
from bs4 import BeautifulSoup
import re

logger = logging.getLogger(__name__)


class ScraperStatus(Enum):
    """Scraper status enumeration."""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    RATE_LIMITED = "rate_limited"


@dataclass
class ScraperConfig:
    """Configuration for web scraper."""
    # Target URLs - ADD YOUR URLS HERE
    base_url: str = ""
    urls: List[str] = field(default_factory=list)
    
    # Rate limiting
    requests_per_minute: int = 30
    min_delay_seconds: float = 1.0
    max_delay_seconds: float = 3.0
    
    # Retry settings
    max_retries: int = 3
    retry_delay_seconds: float = 5.0
    exponential_backoff: bool = True
    
    # Timeout settings
    request_timeout: int = 30
    connect_timeout: int = 10
    
    # Proxy settings (optional)
    proxy_url: Optional[str] = None
    proxy_rotation: bool = False
    proxy_list: List[str] = field(default_factory=list)
    
    # Headers
    custom_headers: Dict[str, str] = field(default_factory=dict)
    rotate_user_agent: bool = True
    
    # Data settings
    cache_enabled: bool = True
    cache_ttl_seconds: int = 300
    
    # Scraper identification
    scraper_name: str = "base_scraper"
    scraper_version: str = "1.0.0"


@dataclass
class ScrapedData:
    """Container for scraped data."""
    url: str
    data: Dict[str, Any]
    scraped_at: datetime
    scraper_name: str
    success: bool
    error_message: Optional[str] = None
    response_status: Optional[int] = None
    response_time_ms: Optional[float] = None
    data_hash: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "url": self.url,
            "data": self.data,
            "scraped_at": self.scraped_at.isoformat(),
            "scraper_name": self.scraper_name,
            "success": self.success,
            "error_message": self.error_message,
            "response_status": self.response_status,
            "response_time_ms": self.response_time_ms,
            "data_hash": self.data_hash
        }


class BaseWebScraper(ABC):
    """
    Base class for all web scrapers.
    
    To create a new scraper:
    1. Inherit from this class
    2. Set your URLs in the config
    3. Implement the parse_page() method
    4. Optionally override other methods for custom behavior
    
    Example:
        class MyScraper(BaseWebScraper):
            def __init__(self):
                config = ScraperConfig(
                    base_url="https://example.com",
                    urls=["https://example.com/page1", "https://example.com/page2"],
                    scraper_name="my_scraper"
                )
                super().__init__(config)
            
            async def parse_page(self, html: str, url: str) -> Dict[str, Any]:
                soup = BeautifulSoup(html, 'html.parser')
                # Extract your data here
                return {"title": soup.title.string}
    """
    
    # Common user agents for rotation
    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0",
        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Mobile/15E148 Safari/604.1",
    ]
    
    def __init__(self, config: ScraperConfig):
        """Initialize the scraper with configuration."""
        self.config = config
        self.status = ScraperStatus.IDLE
        self.session: Optional[aiohttp.ClientSession] = None
        self.request_count = 0
        self.last_request_time: Optional[datetime] = None
        self.cache: Dict[str, ScrapedData] = {}
        self.errors: List[Dict[str, Any]] = []
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_data_scraped": 0,
            "start_time": None,
            "end_time": None
        }
        
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()
    
    async def start(self) -> None:
        """Start the scraper session."""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(
                total=self.config.request_timeout,
                connect=self.config.connect_timeout
            )
            self.session = aiohttp.ClientSession(timeout=timeout)
        self.status = ScraperStatus.RUNNING
        self.stats["start_time"] = datetime.utcnow()
        logger.info(f"Scraper '{self.config.scraper_name}' started")
    
    async def stop(self) -> None:
        """Stop the scraper and close session."""
        if self.session and not self.session.closed:
            await self.session.close()
        self.status = ScraperStatus.IDLE
        self.stats["end_time"] = datetime.utcnow()
        logger.info(f"Scraper '{self.config.scraper_name}' stopped")
    
    def get_headers(self) -> Dict[str, str]:
        """Get request headers with optional user agent rotation."""
        headers = {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Cache-Control": "max-age=0",
        }
        
        # Add user agent
        if self.config.rotate_user_agent:
            headers["User-Agent"] = random.choice(self.USER_AGENTS)
        else:
            headers["User-Agent"] = self.USER_AGENTS[0]
        
        # Add custom headers
        headers.update(self.config.custom_headers)
        
        return headers
    
    def get_proxy(self) -> Optional[str]:
        """Get proxy URL with optional rotation."""
        if self.config.proxy_rotation and self.config.proxy_list:
            return random.choice(self.config.proxy_list)
        return self.config.proxy_url
    
    async def rate_limit(self) -> None:
        """Apply rate limiting between requests."""
        if self.last_request_time:
            elapsed = (datetime.utcnow() - self.last_request_time).total_seconds()
            min_interval = 60.0 / self.config.requests_per_minute
            
            if elapsed < min_interval:
                delay = min_interval - elapsed
                # Add random jitter
                delay += random.uniform(
                    self.config.min_delay_seconds,
                    self.config.max_delay_seconds
                )
                await asyncio.sleep(delay)
        
        self.last_request_time = datetime.utcnow()
    
    def get_cache_key(self, url: str) -> str:
        """Generate cache key for URL."""
        return hashlib.md5(url.encode()).hexdigest()
    
    def get_cached(self, url: str) -> Optional[ScrapedData]:
        """Get cached data if available and not expired."""
        if not self.config.cache_enabled:
            return None
        
        cache_key = self.get_cache_key(url)
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            age = (datetime.utcnow() - cached.scraped_at).total_seconds()
            if age < self.config.cache_ttl_seconds:
                logger.debug(f"Cache hit for {url}")
                return cached
            else:
                del self.cache[cache_key]
        return None
    
    def set_cached(self, url: str, data: ScrapedData) -> None:
        """Cache scraped data."""
        if self.config.cache_enabled:
            cache_key = self.get_cache_key(url)
            self.cache[cache_key] = data
    
    async def fetch_page(self, url: str) -> Optional[str]:
        """
        Fetch a page with retry logic and error handling.
        
        Args:
            url: URL to fetch
            
        Returns:
            HTML content or None if failed
        """
        if not self.session:
            await self.start()
        
        for attempt in range(self.config.max_retries):
            try:
                await self.rate_limit()
                
                start_time = datetime.utcnow()
                
                async with self.session.get(
                    url,
                    headers=self.get_headers(),
                    proxy=self.get_proxy(),
                    ssl=False  # Set to True in production with proper certs
                ) as response:
                    
                    response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                    
                    self.stats["total_requests"] += 1
                    
                    if response.status == 200:
                        html = await response.text()
                        self.stats["successful_requests"] += 1
                        logger.debug(f"Successfully fetched {url} in {response_time:.0f}ms")
                        return html
                    
                    elif response.status == 429:
                        # Rate limited
                        self.status = ScraperStatus.RATE_LIMITED
                        retry_after = int(response.headers.get("Retry-After", 60))
                        logger.warning(f"Rate limited. Waiting {retry_after}s")
                        await asyncio.sleep(retry_after)
                        continue
                    
                    elif response.status >= 500:
                        # Server error, retry
                        logger.warning(f"Server error {response.status} for {url}")
                        
                    else:
                        # Client error, don't retry
                        logger.error(f"Client error {response.status} for {url}")
                        self.stats["failed_requests"] += 1
                        return None
                        
            except asyncio.TimeoutError:
                logger.warning(f"Timeout fetching {url} (attempt {attempt + 1})")
            except aiohttp.ClientError as e:
                logger.warning(f"Client error fetching {url}: {e} (attempt {attempt + 1})")
            except Exception as e:
                logger.error(f"Unexpected error fetching {url}: {e}")
                self.errors.append({
                    "url": url,
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                })
            
            # Exponential backoff
            if attempt < self.config.max_retries - 1:
                delay = self.config.retry_delay_seconds
                if self.config.exponential_backoff:
                    delay *= (2 ** attempt)
                await asyncio.sleep(delay)
        
        self.stats["failed_requests"] += 1
        return None
    
    @abstractmethod
    async def parse_page(self, html: str, url: str) -> Dict[str, Any]:
        """
        Parse HTML and extract data. Override this method in subclasses.
        
        Args:
            html: Raw HTML content
            url: Source URL
            
        Returns:
            Dictionary of extracted data
        """
        pass
    
    async def scrape_url(self, url: str) -> ScrapedData:
        """
        Scrape a single URL.
        
        Args:
            url: URL to scrape
            
        Returns:
            ScrapedData object with results
        """
        # Check cache first
        cached = self.get_cached(url)
        if cached:
            return cached
        
        start_time = datetime.utcnow()
        
        # Fetch page
        html = await self.fetch_page(url)
        
        if html is None:
            return ScrapedData(
                url=url,
                data={},
                scraped_at=datetime.utcnow(),
                scraper_name=self.config.scraper_name,
                success=False,
                error_message="Failed to fetch page"
            )
        
        try:
            # Parse page
            data = await self.parse_page(html, url)
            
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Create data hash for change detection
            data_hash = hashlib.sha256(
                json.dumps(data, sort_keys=True, default=str).encode()
            ).hexdigest()[:16]
            
            result = ScrapedData(
                url=url,
                data=data,
                scraped_at=datetime.utcnow(),
                scraper_name=self.config.scraper_name,
                success=True,
                response_time_ms=response_time,
                data_hash=data_hash
            )
            
            # Cache result
            self.set_cached(url, result)
            
            self.stats["total_data_scraped"] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Error parsing {url}: {e}")
            return ScrapedData(
                url=url,
                data={},
                scraped_at=datetime.utcnow(),
                scraper_name=self.config.scraper_name,
                success=False,
                error_message=str(e)
            )
    
    async def scrape_all(self, urls: Optional[List[str]] = None) -> List[ScrapedData]:
        """
        Scrape all configured URLs or provided list.
        
        Args:
            urls: Optional list of URLs to scrape. If None, uses config.urls
            
        Returns:
            List of ScrapedData objects
        """
        target_urls = urls or self.config.urls
        
        if not target_urls:
            logger.warning("No URLs configured for scraping")
            return []
        
        results = []
        
        async with self:
            for url in target_urls:
                result = await self.scrape_url(url)
                results.append(result)
                
                # Log progress
                success_rate = (self.stats["successful_requests"] / 
                               max(self.stats["total_requests"], 1) * 100)
                logger.info(
                    f"Progress: {len(results)}/{len(target_urls)} URLs "
                    f"({success_rate:.1f}% success rate)"
                )
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get scraper statistics."""
        return {
            **self.stats,
            "scraper_name": self.config.scraper_name,
            "status": self.status.value,
            "cache_size": len(self.cache),
            "error_count": len(self.errors)
        }
    
    async def validate_urls(self) -> Dict[str, bool]:
        """Validate all configured URLs are accessible."""
        results = {}
        
        async with self:
            for url in self.config.urls:
                try:
                    html = await self.fetch_page(url)
                    results[url] = html is not None
                except Exception:
                    results[url] = False
        
        return results


# Utility functions for parsing
def extract_text(soup: BeautifulSoup, selector: str, default: str = "") -> str:
    """Extract text from element using CSS selector."""
    element = soup.select_one(selector)
    return element.get_text(strip=True) if element else default


def extract_texts(soup: BeautifulSoup, selector: str) -> List[str]:
    """Extract text from all matching elements."""
    elements = soup.select(selector)
    return [el.get_text(strip=True) for el in elements]


def extract_attr(soup: BeautifulSoup, selector: str, attr: str, default: str = "") -> str:
    """Extract attribute from element."""
    element = soup.select_one(selector)
    return element.get(attr, default) if element else default


def extract_number(text: str, default: float = 0.0) -> float:
    """Extract number from text."""
    if not text:
        return default
    # Remove common formatting
    cleaned = re.sub(r'[,$%]', '', text)
    match = re.search(r'-?\d+\.?\d*', cleaned)
    return float(match.group()) if match else default


def extract_odds(text: str) -> Optional[int]:
    """Extract American odds from text."""
    if not text:
        return None
    match = re.search(r'([+-]?\d+)', text)
    return int(match.group()) if match else None
