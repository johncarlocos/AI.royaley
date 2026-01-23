"""
LOYALEY - Phase 4 Enterprise Database
Async database connections with connection pooling and health monitoring
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, Optional

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import QueuePool

from app.core.config import settings

logger = logging.getLogger(__name__)

# Base class for SQLAlchemy models
Base = declarative_base()


class DatabaseManager:
    """Enterprise database manager with connection pooling and monitoring"""
    
    def __init__(self):
        self._engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[async_sessionmaker] = None
        self._connection_stats: Dict[str, Any] = {
            "total_connections": 0,
            "active_connections": 0,
            "idle_connections": 0,
            "overflow_connections": 0,
            "queries_executed": 0,
            "errors": 0
        }
    
    async def initialize(self) -> None:
        """Initialize database engine and session factory"""
        if self._engine is not None:
            return
        
        logger.info("Initializing database connection...")
        
        self._engine = create_async_engine(
            settings.DATABASE_URL,
            pool_size=settings.DATABASE_POOL_SIZE,
            max_overflow=settings.DATABASE_MAX_OVERFLOW,
            pool_timeout=settings.DATABASE_POOL_TIMEOUT,
            pool_recycle=settings.DATABASE_POOL_RECYCLE,
            echo=settings.DATABASE_ECHO,
            poolclass=QueuePool,
            pool_pre_ping=True,  # Enable connection health checks
        )
        
        self._session_factory = async_sessionmaker(
            bind=self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autocommit=False,
            autoflush=False
        )
        
        # Set up event listeners for monitoring
        self._setup_event_listeners()
        
        logger.info("Database connection initialized successfully")
    
    def _setup_event_listeners(self) -> None:
        """Set up SQLAlchemy event listeners for monitoring"""
        if not self._engine:
            return
        
        @event.listens_for(self._engine.sync_engine, "checkout")
        def receive_checkout(dbapi_connection, connection_record, connection_proxy):
            self._connection_stats["active_connections"] += 1
        
        @event.listens_for(self._engine.sync_engine, "checkin")
        def receive_checkin(dbapi_connection, connection_record):
            self._connection_stats["active_connections"] -= 1
        
        @event.listens_for(self._engine.sync_engine, "connect")
        def receive_connect(dbapi_connection, connection_record):
            self._connection_stats["total_connections"] += 1
    
    async def close(self) -> None:
        """Close database connection pool"""
        if self._engine:
            logger.info("Closing database connection pool...")
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None
            logger.info("Database connection pool closed")
    
    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get async database session with automatic cleanup"""
        if not self._session_factory:
            await self.initialize()
        
        async with self._session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception as e:
                await session.rollback()
                self._connection_stats["errors"] += 1
                logger.error(f"Database session error: {e}")
                raise
            finally:
                await session.close()
    
    async def execute_raw(self, query: str, params: Optional[Dict] = None) -> Any:
        """Execute raw SQL query"""
        async with self.session() as session:
            result = await session.execute(text(query), params or {})
            self._connection_stats["queries_executed"] += 1
            return result
    
    async def health_check(self) -> Dict[str, Any]:
        """Check database health and connectivity"""
        try:
            start_time = asyncio.get_event_loop().time()
            
            async with self.session() as session:
                result = await session.execute(text("SELECT 1"))
                _ = result.scalar()
            
            latency_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            
            # Get pool statistics
            pool_status = {}
            if self._engine:
                pool = self._engine.pool
                pool_status = {
                    "pool_size": pool.size(),
                    "checked_out": pool.checkedout(),
                    "overflow": pool.overflow(),
                    "checked_in": pool.checkedin()
                }
            
            return {
                "status": "healthy",
                "latency_ms": round(latency_ms, 2),
                "pool": pool_status,
                "stats": self._connection_stats
            }
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "stats": self._connection_stats
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database connection statistics"""
        pool_stats = {}
        if self._engine:
            pool = self._engine.pool
            pool_stats = {
                "pool_size": pool.size(),
                "checked_out": pool.checkedout(),
                "overflow": pool.overflow(),
                "checked_in": pool.checkedin()
            }
        
        return {
            **self._connection_stats,
            "pool": pool_stats
        }


class TransactionManager:
    """Transaction management with savepoints and nested transactions"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    @asynccontextmanager
    async def transaction(self) -> AsyncGenerator[AsyncSession, None]:
        """Execute operations within a transaction"""
        async with self.db_manager.session() as session:
            async with session.begin():
                yield session
    
    @asynccontextmanager
    async def savepoint(self, session: AsyncSession) -> AsyncGenerator[None, None]:
        """Create a savepoint within current transaction"""
        savepoint = await session.begin_nested()
        try:
            yield
        except Exception:
            await savepoint.rollback()
            raise


class QueryBuilder:
    """Fluent query builder for common operations"""
    
    def __init__(self, model: Any):
        self.model = model
        self._filters = []
        self._order_by = []
        self._limit = None
        self._offset = None
    
    def filter(self, *conditions) -> 'QueryBuilder':
        """Add filter conditions"""
        self._filters.extend(conditions)
        return self
    
    def order_by(self, *columns) -> 'QueryBuilder':
        """Add ordering"""
        self._order_by.extend(columns)
        return self
    
    def limit(self, limit: int) -> 'QueryBuilder':
        """Set result limit"""
        self._limit = limit
        return self
    
    def offset(self, offset: int) -> 'QueryBuilder':
        """Set result offset"""
        self._offset = offset
        return self
    
    def build(self):
        """Build SQLAlchemy select statement"""
        from sqlalchemy import select
        
        stmt = select(self.model)
        
        if self._filters:
            stmt = stmt.where(*self._filters)
        
        for col in self._order_by:
            stmt = stmt.order_by(col)
        
        if self._limit:
            stmt = stmt.limit(self._limit)
        
        if self._offset:
            stmt = stmt.offset(self._offset)
        
        return stmt


# Global database manager instance
db_manager = DatabaseManager()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for FastAPI route injection"""
    async with db_manager.session() as session:
        yield session


async def init_db() -> None:
    """Initialize database and create tables"""
    await db_manager.initialize()
    
    # Create minimal required tables for auth if they don't exist
    try:
        async with db_manager._engine.begin() as conn:
            # Import only what we need to avoid FK mismatches across entire schema
            from app.models.models import User  # type: ignore
            await conn.run_sync(User.__table__.create, checkfirst=True)
        logger.info("Minimal auth tables ensured/created")
    except Exception as e:
        logger.error(f"Error creating minimal tables: {e}")


async def close_db() -> None:
    """Close database connections"""
    await db_manager.close()


def get_database_manager() -> DatabaseManager:
    """Get the global database manager instance."""
    return db_manager
