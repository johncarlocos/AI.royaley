"""
ROYALEY - Health Check API Routes
Enterprise-grade health monitoring endpoints
"""

from datetime import datetime
from typing import Optional, List, Dict
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.core.database import get_database_manager, db_manager
from app.core.cache import cache_manager, get_cache_manager
from app.api.dependencies import get_current_user


router = APIRouter(tags=["health"])


# ============================================================================
# SCHEMAS
# ============================================================================

class ComponentHealth(BaseModel):
    name: str
    status: str  # healthy, degraded, unhealthy
    latency_ms: Optional[float] = None
    message: Optional[str] = None
    last_check: datetime


class HealthResponse(BaseModel):
    status: str  # healthy, degraded, unhealthy
    timestamp: datetime
    version: str
    uptime_seconds: float
    components: Dict[str, ComponentHealth]


class DetailedHealthResponse(HealthResponse):
    system: Dict
    database: Dict
    cache: Dict
    ml_models: Dict
    scheduler: Dict
    recent_errors: List[Dict]


class MetricsResponse(BaseModel):
    predictions: Dict
    betting: Dict
    models: Dict
    system: Dict


# ============================================================================
# GLOBAL STATE
# ============================================================================

_start_time = datetime.utcnow()


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.get("", response_model=HealthResponse)
async def basic_health_check():
    """
    Basic health check endpoint for load balancers and monitoring.
    Returns overall system health status.
    """
    from app.core.config import settings
    
    now = datetime.utcnow()
    uptime = (now - _start_time).total_seconds()
    
    components = {}
    overall_status = "healthy"
    
    # Check database
    try:
        db_health_result = await db_manager.health_check()
        db_healthy = db_health_result.get("status") == "healthy" if isinstance(db_health_result, dict) else False
        db_stats = db_manager.get_stats()
        latency = db_health_result.get("latency_ms", 0) if isinstance(db_health_result, dict) else 0
        components["database"] = ComponentHealth(
            name="PostgreSQL",
            status="healthy" if db_healthy else "unhealthy",
            latency_ms=latency,
            last_check=now
        )
        if not db_healthy:
            overall_status = "unhealthy"
    except Exception as e:
        components["database"] = ComponentHealth(
            name="PostgreSQL",
            status="unhealthy",
            message=str(e),
            last_check=now
        )
        overall_status = "unhealthy"
    
    # Check Redis
    try:
        redis_healthy = await cache_manager.health_check()
        components["cache"] = ComponentHealth(
            name="Redis",
            status="healthy" if redis_healthy else "degraded",
            last_check=now
        )
        if not redis_healthy and overall_status == "healthy":
            overall_status = "degraded"
    except Exception as e:
        components["cache"] = ComponentHealth(
            name="Redis",
            status="degraded",
            message=str(e),
            last_check=now
        )
        if overall_status == "healthy":
            overall_status = "degraded"
    
    # Check API
    components["api"] = ComponentHealth(
        name="FastAPI",
        status="healthy",
        latency_ms=1.0,
        last_check=now
    )
    
    return HealthResponse(
        status=overall_status,
        timestamp=now,
        version=settings.APP_VERSION,
        uptime_seconds=uptime,
        components=components
    )


@router.get("/detailed", response_model=DetailedHealthResponse)
async def detailed_health_check(
    current_user: dict = Depends(get_current_user)
):
    """
    Detailed health check with comprehensive system information.
    Requires authentication.
    """
    from app.core.config import settings
    from app.services.monitoring.metrics_service import monitoring_service
    
    now = datetime.utcnow()
    uptime = (now - _start_time).total_seconds()
    
    components = {}
    overall_status = "healthy"
    
    # Database detailed check
    try:
        db_health_result = await db_manager.health_check()
        db_healthy = db_health_result.get("status") == "healthy" if isinstance(db_health_result, dict) else False
        db_stats = db_manager.get_stats()
        db_latency = db_health_result.get("latency_ms", 0) if isinstance(db_health_result, dict) else 0
        
        components["database"] = ComponentHealth(
            name="PostgreSQL",
            status="healthy" if db_healthy else "unhealthy",
            latency_ms=db_latency,
            last_check=now
        )
        
        database_info = {
            "status": "healthy" if db_healthy else "unhealthy",
            "latency_ms": db_latency,
            "pool_size": db_stats.get("pool_size", 0),
            "active_connections": db_stats.get("active", 0),
            "idle_connections": db_stats.get("idle", 0),
            "overflow": db_stats.get("overflow", 0)
        }
        
        if not db_healthy:
            overall_status = "unhealthy"
    except Exception as e:
        components["database"] = ComponentHealth(
            name="PostgreSQL",
            status="unhealthy",
            message=str(e),
            last_check=now
        )
        database_info = {"status": "unhealthy", "error": str(e)}
        overall_status = "unhealthy"
    
    # Cache detailed check
    try:
        redis_health = await cache_manager.health_check()
        cache_stats = await cache_manager.get_stats()
        
        components["cache"] = ComponentHealth(
            name="Redis",
            status="healthy" if redis_health else "degraded",
            last_check=now
        )
        
        cache_info = {
            "status": "healthy" if redis_health else "degraded",
            "hit_rate": cache_stats.get("hit_rate", 0),
            "total_keys": cache_stats.get("total_keys", 0),
            "memory_used_mb": cache_stats.get("memory_used_mb", 0),
            "connected_clients": cache_stats.get("connected_clients", 0)
        }
        
        if not redis_health and overall_status == "healthy":
            overall_status = "degraded"
    except Exception as e:
        components["cache"] = ComponentHealth(
            name="Redis",
            status="degraded",
            message=str(e),
            last_check=now
        )
        cache_info = {"status": "degraded", "error": str(e)}
        if overall_status == "healthy":
            overall_status = "degraded"
    
    # ML Models check
    try:
        from app.services.ml.model_manager import model_manager
        models_status = await model_manager.get_status()
        
        ml_healthy = models_status.get("loaded_models", 0) > 0
        components["ml_models"] = ComponentHealth(
            name="ML Models",
            status="healthy" if ml_healthy else "degraded",
            message=f"{models_status.get('loaded_models', 0)} models loaded",
            last_check=now
        )
        
        ml_models_info = {
            "status": "healthy" if ml_healthy else "degraded",
            "loaded_models": models_status.get("loaded_models", 0),
            "total_models": models_status.get("total_models", 0),
            "last_training": models_status.get("last_training"),
            "average_inference_ms": models_status.get("avg_inference_ms", 0)
        }
    except Exception as e:
        components["ml_models"] = ComponentHealth(
            name="ML Models",
            status="degraded",
            message=str(e),
            last_check=now
        )
        ml_models_info = {"status": "degraded", "error": str(e)}
    
    # Scheduler check
    try:
        from app.services.scheduling.scheduler_service import scheduler_service
        scheduler_status = scheduler_service.get_status()
        
        scheduler_healthy = scheduler_status.get("running", False)
        components["scheduler"] = ComponentHealth(
            name="Scheduler",
            status="healthy" if scheduler_healthy else "degraded",
            message=f"{scheduler_status.get('active_jobs', 0)} active jobs",
            last_check=now
        )
        
        scheduler_info = {
            "status": "healthy" if scheduler_healthy else "degraded",
            "running": scheduler_status.get("running", False),
            "active_jobs": scheduler_status.get("active_jobs", 0),
            "failed_jobs_24h": scheduler_status.get("failed_jobs_24h", 0)
        }
    except Exception as e:
        components["scheduler"] = ComponentHealth(
            name="Scheduler",
            status="degraded",
            message=str(e),
            last_check=now
        )
        scheduler_info = {"status": "degraded", "error": str(e)}
    
    # API component
    components["api"] = ComponentHealth(
        name="FastAPI",
        status="healthy",
        latency_ms=1.0,
        last_check=now
    )
    
    # System metrics
    system_metrics = await monitoring_service.get_system_metrics()
    
    # Recent errors
    recent_errors = await monitoring_service.get_recent_errors(limit=10)
    
    return DetailedHealthResponse(
        status=overall_status,
        timestamp=now,
        version=settings.APP_VERSION,
        uptime_seconds=uptime,
        components=components,
        system=system_metrics,
        database=database_info,
        cache=cache_info,
        ml_models=ml_models_info,
        scheduler=scheduler_info,
        recent_errors=recent_errors
    )


@router.get("/ready")
async def readiness_check():
    """
    Kubernetes readiness probe endpoint.
    Returns 200 if the service is ready to accept traffic.
    """
    # Check critical components
    try:
        db_healthy = await db_manager.health_check()
        if not db_healthy:
            raise HTTPException(status_code=503, detail="Database not ready")
        
        return {"status": "ready", "timestamp": datetime.utcnow().isoformat()}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service not ready: {str(e)}")


@router.get("/live")
async def liveness_check():
    """
    Kubernetes liveness probe endpoint.
    Returns 200 if the service is alive.
    """
    return {"status": "alive", "timestamp": datetime.utcnow().isoformat()}


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics(
    current_user: dict = Depends(get_current_user)
):
    """
    Get application metrics summary.
    """
    from app.services.monitoring.metrics_service import monitoring_service
    
    metrics = await monitoring_service.get_metrics_summary()
    
    return MetricsResponse(
        predictions=metrics.get("predictions", {}),
        betting=metrics.get("betting", {}),
        models=metrics.get("models", {}),
        system=metrics.get("system", {})
    )


@router.get("/prometheus")
async def prometheus_metrics():
    """
    Prometheus metrics endpoint.
    Returns metrics in Prometheus text format.
    """
    from app.services.monitoring.metrics_service import metrics_registry
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    from fastapi.responses import Response
    
    metrics_output = generate_latest(metrics_registry.registry)
    
    return Response(
        content=metrics_output,
        media_type=CONTENT_TYPE_LATEST
    )


@router.get("/components")
async def list_components(
    current_user: dict = Depends(get_current_user)
):
    """
    List all monitored components and their status.
    """
    from app.services.self_healing.self_healing_service import self_healing_service
    
    components = self_healing_service.get_all_component_status()
    
    return {
        "components": components,
        "timestamp": datetime.utcnow().isoformat()
    }


@router.post("/components/{component_name}/restart")
async def restart_component(
    component_name: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Manually trigger component restart.
    Requires admin role.
    """
    if current_user.get("role") not in ["admin", "system"]:
        raise HTTPException(
            status_code=403,
            detail="Admin access required"
        )
    
    from app.services.self_healing.self_healing_service import self_healing_service
    
    try:
        result = await self_healing_service.manual_recovery(component_name)
        return {
            "status": "success" if result else "failed",
            "component": component_name,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to restart component: {str(e)}"
        )


@router.get("/alerts")
async def get_active_alerts(
    current_user: dict = Depends(get_current_user)
):
    """
    Get active system alerts.
    """
    from app.services.alerting.alerting_service import alerting_service
    
    alerts = alerting_service.get_recent_alerts(limit=50)
    active = [a for a in alerts if not a.get("resolved", False)]
    
    return {
        "active_alerts": len(active),
        "alerts": active,
        "timestamp": datetime.utcnow().isoformat()
    }
