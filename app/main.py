"""
ROYALEY - Main FastAPI Application
Phase 4: Enterprise Features

Enterprise-grade FastAPI application with:
- Complete middleware stack
- CORS configuration
- Rate limiting
- Request tracing
- Error handling
- Health monitoring
- Metrics collection
"""

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional, Callable

from fastapi import FastAPI, Request, Response, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.middleware.base import BaseHTTPMiddleware
import uvicorn

from app.core.config import get_settings
from app.core.database import get_database_manager, init_db
from app.core.cache import get_cache_manager
from app.services.monitoring import get_monitoring_service
from app.services.alerting import get_alerting_service, AlertSeverity
from app.services.self_healing import get_self_healing_service
from app.services.scheduling import get_scheduler_service

from app.api.routes import (
    auth_router,
    predictions_router,
    games_router,
    odds_router,
    betting_router,
    models_router,
    health_router,
    analytics_router,
    monitoring_router,
    admin_router,
    backtest_router,
    player_props_router,
)
from app.api.routes.reports import router as reports_router
from app.api.routes.predictions_public import router as public_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

settings = get_settings()


# ============================================================================
# Lifespan Management
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup and shutdown events.
    """
    logger.info("=" * 60)
    logger.info("ROYALEY - Phase 4 Enterprise Platform")
    logger.info("=" * 60)
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Debug Mode: {settings.debug}")
    logger.info("Starting up...")
    
    # Initialize core services
    try:
        # Database
        db_manager = get_database_manager()
        await db_manager.initialize()
        logger.info("✓ Database initialized")
        # Ensure tables exist
        await init_db()
        logger.info("✓ Database tables ensured")
        
        # Cache
        cache_manager = get_cache_manager()
        await cache_manager.initialize()
        logger.info("✓ Redis cache initialized")
        
        # Monitoring
        monitoring_service = get_monitoring_service()
        logger.info("✓ Monitoring service initialized")
        
        # Alerting
        alerting_service = get_alerting_service()
        logger.info("✓ Alerting service initialized")
        
        # Self-healing
        self_healing_service = get_self_healing_service()
        await self_healing_service.start()
        logger.info("✓ Self-healing service started")
        
        # Scheduler
        scheduler_service = get_scheduler_service()
        await scheduler_service.initialize()
        await scheduler_service.start()
        logger.info("✓ Scheduler service started")
        
        # Send startup notification
        await alerting_service.info(
            "System Startup",
            f"Royaley Phase 4 started successfully in {settings.environment} mode"
        )
        
        logger.info("=" * 60)
        logger.info("All services initialized successfully!")
        logger.info(f"API available at: http://0.0.0.0:{settings.port}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    
    try:
        # Stop scheduler
        scheduler_service = get_scheduler_service()
        await scheduler_service.stop()
        logger.info("✓ Scheduler stopped")
        
        # Stop self-healing
        self_healing_service = get_self_healing_service()
        await self_healing_service.stop()
        logger.info("✓ Self-healing stopped")
        
        # Close cache
        cache_manager = get_cache_manager()
        await cache_manager.close()
        logger.info("✓ Cache closed")
        
        # Close database
        db_manager = get_database_manager()
        await db_manager.close()
        logger.info("✓ Database closed")
        
        # Send shutdown notification
        alerting_service = get_alerting_service()
        await alerting_service.info(
            "System Shutdown",
            "Royaley Phase 4 shutting down gracefully"
        )
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")
    
    logger.info("Shutdown complete")


# ============================================================================
# Create FastAPI Application
# ============================================================================

app = FastAPI(
    title=settings.app_name,
    description="Enterprise-Grade Sports Prediction Platform - Phase 4",
    version=settings.app_version,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    openapi_url="/openapi.json" if settings.debug else None,
    lifespan=lifespan,
)


# ============================================================================
# Middleware
# ============================================================================

class RequestTrackingMiddleware(BaseHTTPMiddleware):
    """Add request tracking and timing."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Record start time
        start_time = time.perf_counter()
        
        # Get monitoring service
        try:
            monitoring = get_monitoring_service()
        except Exception:
            monitoring = None
        
        try:
            response = await call_next(request)
            
            # Calculate duration
            duration = time.perf_counter() - start_time
            
            # Record metrics
            if monitoring:
                monitoring.record_http_request(
                    method=request.method,
                    endpoint=request.url.path,
                    status_code=response.status_code,
                    duration=duration
                )
            
            # Add headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = f"{duration:.4f}s"
            
            return response
            
        except Exception as e:
            # Record error
            duration = time.perf_counter() - start_time
            
            if monitoring:
                monitoring.record_http_request(
                    method=request.method,
                    endpoint=request.url.path,
                    status_code=500,
                    duration=duration
                )
                monitoring.record_error("unhandled_exception", "api")
            
            logger.error(f"Unhandled error in request {request_id}: {e}")
            raise


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware."""
    
    def __init__(self, app, requests_per_minute: int = 100):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests: dict = {}  # IP -> list of timestamps
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/api/v1/health", "/metrics"]:
            return await call_next(request)
        
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        
        # Check rate limit
        current_time = time.time()
        window_start = current_time - 60  # 1 minute window
        
        if client_ip not in self.requests:
            self.requests[client_ip] = []
        
        # Clean old requests
        self.requests[client_ip] = [
            ts for ts in self.requests[client_ip] 
            if ts > window_start
        ]
        
        # Check limit
        if len(self.requests[client_ip]) >= self.requests_per_minute:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "detail": f"Maximum {self.requests_per_minute} requests per minute",
                    "retry_after": 60
                },
                headers={"Retry-After": "60"}
            )
        
        # Record request
        self.requests[client_ip].append(current_time)
        
        return await call_next(request)


# Add middleware (order matters - first added is outermost)
app.add_middleware(GZipMiddleware, minimum_size=1000)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Response-Time"],
)

app.add_middleware(RateLimitMiddleware, requests_per_minute=settings.rate_limit_requests)
app.add_middleware(RequestTrackingMiddleware)


# ============================================================================
# Exception Handlers
# ============================================================================

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors."""
    errors = []
    for error in exc.errors():
        errors.append({
            "field": ".".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "type": error["type"]
        })
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation Error",
            "detail": errors,
            "request_id": getattr(request.state, "request_id", None)
        }
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "request_id": getattr(request.state, "request_id", None)
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle all other exceptions."""
    request_id = getattr(request.state, "request_id", None)
    logger.error(f"Unhandled exception [{request_id}]: {exc}", exc_info=True)
    
    # Record error metric
    try:
        monitoring = get_monitoring_service()
        monitoring.record_error("unhandled_exception", "api")
    except Exception:
        pass
    
    # Send alert for critical errors
    try:
        alerting = get_alerting_service()
        await alerting.error(
            "Unhandled Exception",
            f"Request {request_id}: {str(exc)[:500]}"
        )
    except Exception:
        pass
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "detail": "An unexpected error occurred" if not settings.debug else str(exc),
            "request_id": request_id
        }
    )


# ============================================================================
# Include Routers
# ============================================================================

# API v1 routes
API_V1_PREFIX = "/api/v1"

# Root API v1 endpoint
@app.get(API_V1_PREFIX)
async def api_v1_root():
    """API v1 root endpoint with available routes."""
    from app.core.config import settings
    return {
        "version": "v1",
        "name": settings.app_name,
        "status": "operational",
        "endpoints": {
            "health": f"{API_V1_PREFIX}/health",
            "auth": f"{API_V1_PREFIX}/auth",
            "predictions": f"{API_V1_PREFIX}/predictions",
            "games": f"{API_V1_PREFIX}/games",
            "odds": f"{API_V1_PREFIX}/odds",
            "betting": f"{API_V1_PREFIX}/betting",
            "analytics": f"{API_V1_PREFIX}/analytics",
            "reports": f"{API_V1_PREFIX}/reports",
            "models": f"{API_V1_PREFIX}/models",
            "player-props": f"{API_V1_PREFIX}/player-props",
            "backtest": f"{API_V1_PREFIX}/backtest",
            "monitoring": f"{API_V1_PREFIX}/monitoring",
            "admin": f"{API_V1_PREFIX}/admin",
        },
        "docs": f"{API_V1_PREFIX}/docs" if settings.debug else None
    }

app.include_router(
    health_router,
    prefix=f"{API_V1_PREFIX}/health",
    tags=["Health"]
)

app.include_router(
    auth_router,
    prefix=f"{API_V1_PREFIX}/auth",
    tags=["Authentication"]
)

app.include_router(
    predictions_router,
    prefix=f"{API_V1_PREFIX}/predictions",
    tags=["Predictions"]
)

app.include_router(
    games_router,
    prefix=f"{API_V1_PREFIX}/games",
    tags=["Games"]
)

app.include_router(
    odds_router,
    prefix=f"{API_V1_PREFIX}/odds",
    tags=["Odds"]
)

app.include_router(
    betting_router,
    prefix=f"{API_V1_PREFIX}/betting",
    tags=["Betting"]
)

app.include_router(
    models_router,
    prefix=f"{API_V1_PREFIX}/models",
    tags=["ML Models"]
)

app.include_router(
    player_props_router,
    prefix=f"{API_V1_PREFIX}/player-props",
    tags=["Player Props"]
)

app.include_router(
    analytics_router,
    prefix=f"{API_V1_PREFIX}/analytics",
    tags=["Analytics"]
)

app.include_router(
    backtest_router,
    prefix=f"{API_V1_PREFIX}/backtest",
    tags=["Backtesting"]
)

app.include_router(
    monitoring_router,
    prefix=f"{API_V1_PREFIX}/monitoring",
    tags=["Monitoring"]
)

app.include_router(
    admin_router,
    prefix=f"{API_V1_PREFIX}/admin",
    tags=["Admin"]
)

app.include_router(
    reports_router,
    prefix=f"{API_V1_PREFIX}/reports",
    tags=["Reports"]
)

# Public API (no auth required - for frontend live data)
app.include_router(
    public_router,
    prefix=f"{API_V1_PREFIX}/public",
    tags=["Public"]
)


# ============================================================================
# Root Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "phase": "4 - Enterprise Features",
        "status": "operational",
        "docs": "/docs" if settings.debug else None,
        "health": "/api/v1/health"
    }


@app.get("/health")
async def health():
    """Simple health check endpoint."""
    return {"status": "healthy"}


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    try:
        monitoring = get_monitoring_service()
        return Response(
            content=monitoring.metrics.get_metrics(),
            media_type=monitoring.metrics.get_content_type()
        )
    except Exception as e:
        logger.error(f"Error exporting metrics: {e}")
        return Response(
            content="# Error exporting metrics",
            media_type="text/plain",
            status_code=500
        )


# ============================================================================
# Run Application
# ============================================================================

def run():
    """Run the FastAPI application."""
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.port,
        reload=settings.debug,
        workers=settings.workers if not settings.debug else 1,
        log_level="info" if not settings.debug else "debug",
        access_log=True,
    )


if __name__ == "__main__":
    run()