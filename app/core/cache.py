"""
ROYALEY - Phase 4 Enterprise Cache
Redis caching with distributed locking, pub/sub, and circuit breaker
"""

import asyncio
import json
import logging
import pickle
import time
from contextlib import asynccontextmanager
from datetime import timedelta
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Union

import redis.asyncio as redis
from redis.asyncio.lock import Lock
from redis.exceptions import ConnectionError, TimeoutError

from app.core.config import settings

logger = logging.getLogger(__name__)


class CachePrefix(str, Enum):
    """Cache key prefixes for organization"""
    PREDICTIONS = "pred"
    ODDS = "odds"
    GAMES = "games"
    MODELS = "models"
    FEATURES = "features"
    USER = "user"
    SESSION = "session"
    RATE_LIMIT = "rate"
    LOCK = "lock"
    METRICS = "metrics"


class CircuitBreakerState(str, Enum):
    """Circuit breaker states"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Circuit breaker for cache resilience"""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 30,
        half_open_requests: int = 3
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_requests = half_open_requests
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0
        self.half_open_successes = 0
    
    def can_execute(self) -> bool:
        """Check if request can be executed"""
        if self.state == CircuitBreakerState.CLOSED:
            return True
        
        if self.state == CircuitBreakerState.OPEN:
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = CircuitBreakerState.HALF_OPEN
                self.half_open_successes = 0
                return True
            return False
        
        # HALF_OPEN state
        return True
    
    def record_success(self):
        """Record successful execution"""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.half_open_successes += 1
            if self.half_open_successes >= self.half_open_requests:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
        else:
            self.failure_count = 0
    
    def record_failure(self):
        """Record failed execution"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.OPEN
        elif self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN


class CacheManager:
    """Enterprise Redis cache manager"""
    
    def __init__(self):
        self._client: Optional[redis.Redis] = None
        self._pubsub: Optional[redis.client.PubSub] = None
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=settings.CIRCUIT_BREAKER_THRESHOLD,
            recovery_timeout=settings.CIRCUIT_BREAKER_TIMEOUT
        )
        self._stats = {
            "hits": 0,
            "misses": 0,
            "errors": 0,
            "sets": 0
        }
    
    async def initialize(self) -> None:
        """Initialize Redis connection"""
        if self._client is not None:
            return
        
        logger.info("Initializing Redis connection...")
        
        self._client = redis.from_url(
            settings.REDIS_URL,
            max_connections=settings.REDIS_MAX_CONNECTIONS,
            socket_timeout=settings.REDIS_SOCKET_TIMEOUT,
            decode_responses=False,  # Handle encoding manually for flexibility
            retry_on_timeout=True
        )
        
        # Test connection
        try:
            await self._client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self._client = None
            raise
    
    async def close(self) -> None:
        """Close Redis connection"""
        if self._client:
            logger.info("Closing Redis connection...")
            if self._pubsub:
                await self._pubsub.close()
            await self._client.close()
            self._client = None
            logger.info("Redis connection closed")
    
    def _make_key(self, prefix: CachePrefix, key: str) -> str:
        """Create prefixed cache key"""
        return f"{settings.APP_NAME}:{prefix.value}:{key}"
    
    async def _execute_with_circuit_breaker(self, operation: Callable) -> Any:
        """Execute operation with circuit breaker protection"""
        if not self._circuit_breaker.can_execute():
            self._stats["errors"] += 1
            raise ConnectionError("Circuit breaker is open")
        
        try:
            result = await operation()
            self._circuit_breaker.record_success()
            return result
        except (ConnectionError, TimeoutError) as e:
            self._circuit_breaker.record_failure()
            self._stats["errors"] += 1
            logger.error(f"Redis operation failed: {e}")
            raise
    
    async def get(
        self,
        key: str,
        prefix: CachePrefix = CachePrefix.PREDICTIONS,
        deserialize: bool = True
    ) -> Optional[Any]:
        """Get value from cache"""
        if not self._client:
            await self.initialize()
        
        full_key = self._make_key(prefix, key)
        
        async def _get():
            return await self._client.get(full_key)
        
        try:
            value = await self._execute_with_circuit_breaker(_get)
            
            if value is None:
                self._stats["misses"] += 1
                return None
            
            self._stats["hits"] += 1
            
            if deserialize:
                try:
                    return json.loads(value.decode('utf-8'))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    return pickle.loads(value)
            
            return value
        except Exception as e:
            logger.warning(f"Cache get failed for {full_key}: {e}")
            return None
    
    async def set(
        self,
        key: str,
        value: Any,
        prefix: CachePrefix = CachePrefix.PREDICTIONS,
        ttl: Optional[int] = None,
        serialize: bool = True
    ) -> bool:
        """Set value in cache"""
        if not self._client:
            await self.initialize()
        
        full_key = self._make_key(prefix, key)
        ttl = ttl or settings.CACHE_TTL_DEFAULT
        
        if serialize:
            try:
                serialized = json.dumps(value).encode('utf-8')
            except (TypeError, ValueError):
                serialized = pickle.dumps(value)
        else:
            serialized = value
        
        async def _set():
            return await self._client.setex(full_key, ttl, serialized)
        
        try:
            await self._execute_with_circuit_breaker(_set)
            self._stats["sets"] += 1
            return True
        except Exception as e:
            logger.warning(f"Cache set failed for {full_key}: {e}")
            return False
    
    async def delete(
        self,
        key: str,
        prefix: CachePrefix = CachePrefix.PREDICTIONS
    ) -> bool:
        """Delete key from cache"""
        if not self._client:
            return False
        
        full_key = self._make_key(prefix, key)
        
        async def _delete():
            return await self._client.delete(full_key)
        
        try:
            await self._execute_with_circuit_breaker(_delete)
            return True
        except Exception:
            return False
    
    async def delete_pattern(
        self,
        pattern: str,
        prefix: CachePrefix = CachePrefix.PREDICTIONS
    ) -> int:
        """Delete all keys matching pattern"""
        if not self._client:
            return 0
        
        full_pattern = self._make_key(prefix, pattern)
        deleted = 0
        
        try:
            async for key in self._client.scan_iter(match=full_pattern):
                await self._client.delete(key)
                deleted += 1
            return deleted
        except Exception as e:
            logger.warning(f"Pattern delete failed for {full_pattern}: {e}")
            return deleted
    
    async def exists(
        self,
        key: str,
        prefix: CachePrefix = CachePrefix.PREDICTIONS
    ) -> bool:
        """Check if key exists"""
        if not self._client:
            return False
        
        full_key = self._make_key(prefix, key)
        
        try:
            return bool(await self._client.exists(full_key))
        except Exception:
            return False
    
    async def increment(
        self,
        key: str,
        prefix: CachePrefix = CachePrefix.METRICS,
        amount: int = 1
    ) -> int:
        """Increment counter"""
        if not self._client:
            await self.initialize()
        
        full_key = self._make_key(prefix, key)
        
        try:
            return await self._client.incrby(full_key, amount)
        except Exception:
            return 0
    
    async def get_many(
        self,
        keys: List[str],
        prefix: CachePrefix = CachePrefix.PREDICTIONS
    ) -> Dict[str, Any]:
        """Get multiple values"""
        if not self._client or not keys:
            return {}
        
        full_keys = [self._make_key(prefix, k) for k in keys]
        
        try:
            values = await self._client.mget(full_keys)
            result = {}
            for key, value in zip(keys, values):
                if value is not None:
                    try:
                        result[key] = json.loads(value.decode('utf-8'))
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        result[key] = pickle.loads(value)
            return result
        except Exception as e:
            logger.warning(f"Batch get failed: {e}")
            return {}
    
    async def set_many(
        self,
        items: Dict[str, Any],
        prefix: CachePrefix = CachePrefix.PREDICTIONS,
        ttl: Optional[int] = None
    ) -> bool:
        """Set multiple values"""
        if not self._client or not items:
            return False
        
        ttl = ttl or settings.CACHE_TTL_DEFAULT
        
        try:
            async with self._client.pipeline(transaction=True) as pipe:
                for key, value in items.items():
                    full_key = self._make_key(prefix, key)
                    try:
                        serialized = json.dumps(value).encode('utf-8')
                    except (TypeError, ValueError):
                        serialized = pickle.dumps(value)
                    pipe.setex(full_key, ttl, serialized)
                await pipe.execute()
            return True
        except Exception as e:
            logger.warning(f"Batch set failed: {e}")
            return False
    
    @asynccontextmanager
    async def lock(
        self,
        name: str,
        timeout: int = 10,
        blocking_timeout: float = 5
    ):
        """Distributed lock context manager"""
        if not self._client:
            await self.initialize()
        
        lock_key = self._make_key(CachePrefix.LOCK, name)
        lock = Lock(
            self._client,
            lock_key,
            timeout=timeout,
            blocking_timeout=blocking_timeout
        )
        
        try:
            acquired = await lock.acquire()
            if not acquired:
                raise TimeoutError(f"Could not acquire lock: {name}")
            yield lock
        finally:
            if lock.locked():
                await lock.release()
    
    async def publish(self, channel: str, message: Any) -> int:
        """Publish message to channel"""
        if not self._client:
            await self.initialize()
        
        try:
            serialized = json.dumps(message)
            return await self._client.publish(channel, serialized)
        except Exception as e:
            logger.warning(f"Publish failed: {e}")
            return 0
    
    async def subscribe(self, *channels: str):
        """Subscribe to channels"""
        if not self._client:
            await self.initialize()
        
        if not self._pubsub:
            self._pubsub = self._client.pubsub()
        
        await self._pubsub.subscribe(*channels)
        return self._pubsub
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Redis health"""
        try:
            if not self._client:
                return {"status": "disconnected"}
            
            start = time.time()
            await self._client.ping()
            latency = (time.time() - start) * 1000
            
            info = await self._client.info("memory")
            
            return {
                "status": "healthy",
                "latency_ms": round(latency, 2),
                "memory_used": info.get("used_memory_human", "unknown"),
                "connected_clients": info.get("connected_clients", 0),
                "circuit_breaker": self._circuit_breaker.state.value,
                "stats": self._stats
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "circuit_breaker": self._circuit_breaker.state.value
            }
    
    async def clear_all(self) -> bool:
        """Clear all cache (use with caution!)"""
        if not self._client:
            return False
        
        try:
            await self._client.flushdb()
            return True
        except Exception:
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = (self._stats["hits"] / total * 100) if total > 0 else 0
        
        return {
            **self._stats,
            "total_requests": total,
            "hit_rate_percent": round(hit_rate, 2),
            "circuit_breaker_state": self._circuit_breaker.state.value
        }


def cached(
    prefix: CachePrefix = CachePrefix.PREDICTIONS,
    ttl: int = 300,
    key_builder: Optional[Callable] = None
):
    """Decorator for caching function results"""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Build cache key
            if key_builder:
                cache_key = key_builder(*args, **kwargs)
            else:
                key_parts = [func.__name__]
                key_parts.extend(str(a) for a in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = ":".join(key_parts)
            
            # Try to get from cache
            result = await cache_manager.get(cache_key, prefix)
            if result is not None:
                return result
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            await cache_manager.set(cache_key, result, prefix, ttl)
            
            return result
        return wrapper
    return decorator


# Global cache manager instance
cache_manager = CacheManager()


async def init_cache() -> None:
    """Initialize cache"""
    await cache_manager.initialize()


async def close_cache() -> None:
    """Close cache connection"""
    await cache_manager.close()


def get_cache_manager() -> CacheManager:
    """Get the global cache manager instance."""
    return cache_manager
