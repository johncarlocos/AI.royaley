"""
LOYALEY - Monitoring API Routes
Enterprise-grade system monitoring and metrics endpoints
"""

from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_

from app.core.database import get_db
from app.core.security import security_manager
from app.api.dependencies import require_roles
from app.models.models import (
    SystemHealthSnapshot, Alert, ScheduledTask, DataQualityCheck,
    AlertSeverity, HealthStatus, TaskStatus
)
from app.services.monitoring import get_monitoring_service, ComponentHealth
from app.services.alerting import get_alerting_service
from app.services.self_healing import get_self_healing_service
from app.services.scheduling import get_scheduler_service

router = APIRouter()


# ============================================================================
# Response Schemas
# ============================================================================

class SystemHealthResponse(BaseModel):
    """System health status"""
    status: str
    overall_score: float
    components: Dict[str, Any]
    last_check: datetime
    uptime_hours: float


class ComponentHealthResponse(BaseModel):
    """Individual component health"""
    name: str
    status: str
    latency_ms: float
    last_check: datetime
    details: Dict[str, Any]


class AlertResponse(BaseModel):
    """Alert information"""
    id: int
    severity: str
    title: str
    message: str
    source: str
    created_at: datetime
    acknowledged: bool
    acknowledged_at: Optional[datetime]


class MetricSummary(BaseModel):
    """Metrics summary"""
    name: str
    value: float
    unit: str
    trend: str  # 'up', 'down', 'stable'
    change_pct: float


class SchedulerStatus(BaseModel):
    """Scheduler status"""
    running: bool
    total_jobs: int
    active_jobs: int
    failed_jobs_24h: int
    next_run: Optional[datetime]
    jobs: List[Dict[str, Any]]


class CircuitBreakerStatus(BaseModel):
    """Circuit breaker status"""
    service: str
    state: str  # 'closed', 'open', 'half_open'
    failure_count: int
    last_failure: Optional[datetime]
    last_success: Optional[datetime]


class DataQualityReport(BaseModel):
    """Data quality report"""
    overall_score: float
    checks_passed: int
    checks_failed: int
    last_check: datetime
    issues: List[Dict[str, Any]]


# ============================================================================
# Health Monitoring Endpoints
# ============================================================================

@router.get("/health", response_model=SystemHealthResponse)
async def get_system_health(
    current_user = Depends(require_roles(["admin"]))
):
    """
    Get comprehensive system health status.
    
    Returns health of all system components including database, cache, ML models, etc.
    """
    monitoring = get_monitoring_service()
    health = await monitoring.get_system_health()
    
    return SystemHealthResponse(
        status=health.get("status", "unknown"),
        overall_score=health.get("score", 0.0),
        components=health.get("components", {}),
        last_check=datetime.utcnow(),
        uptime_hours=health.get("uptime_hours", 0.0)
    )


@router.get("/health/components", response_model=List[ComponentHealthResponse])
async def get_component_health(
    current_user = Depends(require_roles(["admin"]))
):
    """
    Get health status of individual components.
    
    Shows detailed health information for each system component.
    """
    monitoring = get_monitoring_service()
    components = await monitoring.get_all_component_health()
    
    return [
        ComponentHealthResponse(
            name=c.name,
            status=c.status.value if hasattr(c.status, 'value') else str(c.status),
            latency_ms=c.latency_ms,
            last_check=c.last_check,
            details=c.details or {}
        )
        for c in components
    ]


@router.get("/health/history")
async def get_health_history(
    hours: int = Query(default=24, ge=1, le=168),
    db: AsyncSession = Depends(get_db),
    current_user = Depends(require_roles(["admin"]))
):
    """
    Get historical health snapshots.
    
    Shows system health over time for trend analysis.
    """
    start_time = datetime.utcnow() - timedelta(hours=hours)
    
    query = select(SystemHealthSnapshot).where(
        SystemHealthSnapshot.captured_at >= start_time
    ).order_by(SystemHealthSnapshot.captured_at.desc())
    
    result = await db.execute(query)
    snapshots = result.scalars().all()
    
    return {
        "total_snapshots": len(snapshots),
        "time_range_hours": hours,
        "snapshots": [
            {
                "captured_at": s.captured_at.isoformat(),
                "overall_status": s.overall_status.value if s.overall_status else "unknown",
                "cpu_usage": s.cpu_usage,
                "memory_usage": s.memory_usage,
                "disk_usage": s.disk_usage,
                "api_latency_p50": s.api_latency_p50,
                "api_latency_p99": s.api_latency_p99,
                "active_connections": s.active_connections
            }
            for s in snapshots[:100]
        ]
    }


# ============================================================================
# Alerts Endpoints
# ============================================================================

@router.get("/alerts", response_model=List[AlertResponse])
async def get_alerts(
    severity: Optional[str] = Query(default=None),
    acknowledged: Optional[bool] = Query(default=None),
    hours: int = Query(default=24, ge=1, le=168),
    limit: int = Query(default=50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
    current_user = Depends(require_roles(["admin"]))
):
    """
    Get system alerts.
    
    Returns recent alerts with optional filtering by severity and acknowledgment status.
    """
    start_time = datetime.utcnow() - timedelta(hours=hours)
    
    query = select(Alert).where(Alert.created_at >= start_time)
    
    if severity:
        try:
            sev_enum = AlertSeverity(severity.upper())
            query = query.where(Alert.severity == sev_enum)
        except ValueError:
            pass
    
    if acknowledged is not None:
        query = query.where(Alert.acknowledged == acknowledged)
    
    query = query.order_by(Alert.created_at.desc()).limit(limit)
    
    result = await db.execute(query)
    alerts = result.scalars().all()
    
    return [
        AlertResponse(
            id=a.id,
            severity=a.severity.value if a.severity else "info",
            title=a.title,
            message=a.message,
            source=a.source or "system",
            created_at=a.created_at,
            acknowledged=a.acknowledged or False,
            acknowledged_at=a.acknowledged_at
        )
        for a in alerts
    ]


@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: int,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(require_roles(["admin"]))
):
    """
    Acknowledge an alert.
    
    Marks the alert as acknowledged by the current user.
    """
    query = select(Alert).where(Alert.id == alert_id)
    result = await db.execute(query)
    alert = result.scalar_one_or_none()
    
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    
    alert.acknowledged = True
    alert.acknowledged_at = datetime.utcnow()
    alert.acknowledged_by = current_user.id
    
    await db.commit()
    
    return {"message": "Alert acknowledged", "alert_id": alert_id}


@router.get("/alerts/summary")
async def get_alerts_summary(
    hours: int = Query(default=24, ge=1, le=168),
    db: AsyncSession = Depends(get_db),
    current_user = Depends(require_roles(["admin"]))
):
    """
    Get alerts summary.
    
    Returns count of alerts by severity and acknowledgment status.
    """
    start_time = datetime.utcnow() - timedelta(hours=hours)
    
    query = select(Alert).where(Alert.created_at >= start_time)
    result = await db.execute(query)
    alerts = result.scalars().all()
    
    by_severity = {}
    for sev in AlertSeverity:
        by_severity[sev.value] = len([a for a in alerts if a.severity == sev])
    
    unacknowledged = len([a for a in alerts if not a.acknowledged])
    
    return {
        "total": len(alerts),
        "unacknowledged": unacknowledged,
        "by_severity": by_severity,
        "time_range_hours": hours
    }


# ============================================================================
# Metrics Endpoints
# ============================================================================

@router.get("/metrics")
async def get_metrics(
    current_user = Depends(require_roles(["admin"]))
):
    """
    Get current system metrics.
    
    Returns real-time metrics from the monitoring service.
    """
    monitoring = get_monitoring_service()
    metrics = await monitoring.get_current_metrics()
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "metrics": metrics
    }


@router.get("/metrics/prometheus")
async def get_prometheus_metrics(
    current_user = Depends(require_roles(["admin"]))
):
    """
    Get metrics in Prometheus format.
    
    Returns metrics formatted for Prometheus scraping.
    """
    monitoring = get_monitoring_service()
    return await monitoring.get_prometheus_metrics()


@router.get("/metrics/summary", response_model=List[MetricSummary])
async def get_metrics_summary(
    current_user = Depends(require_roles(["admin"]))
):
    """
    Get summary of key metrics.
    
    Returns key metrics with trend information.
    """
    monitoring = get_monitoring_service()
    summary = await monitoring.get_metrics_summary()
    
    return [
        MetricSummary(
            name=m["name"],
            value=m["value"],
            unit=m.get("unit", ""),
            trend=m.get("trend", "stable"),
            change_pct=m.get("change_pct", 0.0)
        )
        for m in summary
    ]


# ============================================================================
# Scheduler Endpoints
# ============================================================================

@router.get("/scheduler/status", response_model=SchedulerStatus)
async def get_scheduler_status(
    db: AsyncSession = Depends(get_db),
    current_user = Depends(require_roles(["admin"]))
):
    """
    Get scheduler status.
    
    Shows status of background job scheduler and scheduled tasks.
    """
    scheduler = get_scheduler_service()
    status = await scheduler.get_status()
    
    # Get tasks from database
    query = select(ScheduledTask).where(ScheduledTask.is_active == True)
    result = await db.execute(query)
    tasks = result.scalars().all()
    
    # Count failed in last 24h
    yesterday = datetime.utcnow() - timedelta(hours=24)
    failed_query = select(func.count(ScheduledTask.id)).where(
        and_(
            ScheduledTask.last_run >= yesterday,
            ScheduledTask.last_status == TaskStatus.FAILED
        )
    )
    failed_result = await db.execute(failed_query)
    failed_count = failed_result.scalar() or 0
    
    return SchedulerStatus(
        running=status.get("running", False),
        total_jobs=len(tasks),
        active_jobs=len([t for t in tasks if t.is_active]),
        failed_jobs_24h=failed_count,
        next_run=status.get("next_run"),
        jobs=[
            {
                "name": t.name,
                "schedule": t.schedule,
                "last_run": t.last_run.isoformat() if t.last_run else None,
                "next_run": t.next_run.isoformat() if t.next_run else None,
                "status": t.last_status.value if t.last_status else "unknown"
            }
            for t in tasks[:20]
        ]
    )


@router.post("/scheduler/jobs/{job_name}/run")
async def trigger_job(
    job_name: str,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(require_roles(["admin"]))
):
    """
    Manually trigger a scheduled job.
    
    Immediately executes the specified job regardless of schedule.
    """
    scheduler = get_scheduler_service()
    
    try:
        await scheduler.run_job(job_name)
        return {"message": f"Job '{job_name}' triggered successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/scheduler/jobs/{job_name}/pause")
async def pause_job(
    job_name: str,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(require_roles(["admin"]))
):
    """
    Pause a scheduled job.
    
    Prevents the job from running until resumed.
    """
    query = select(ScheduledTask).where(ScheduledTask.name == job_name)
    result = await db.execute(query)
    task = result.scalar_one_or_none()
    
    if not task:
        raise HTTPException(status_code=404, detail="Job not found")
    
    task.is_active = False
    await db.commit()
    
    return {"message": f"Job '{job_name}' paused"}


# ============================================================================
# Self-Healing Endpoints
# ============================================================================

@router.get("/circuit-breakers", response_model=List[CircuitBreakerStatus])
async def get_circuit_breakers(
    current_user = Depends(require_roles(["admin"]))
):
    """
    Get circuit breaker status for all services.
    
    Shows which services have tripped circuit breakers.
    """
    healing = get_self_healing_service()
    breakers = await healing.get_circuit_breaker_status()
    
    return [
        CircuitBreakerStatus(
            service=b["service"],
            state=b["state"],
            failure_count=b["failure_count"],
            last_failure=b.get("last_failure"),
            last_success=b.get("last_success")
        )
        for b in breakers
    ]


@router.post("/circuit-breakers/{service}/reset")
async def reset_circuit_breaker(
    service: str,
    current_user = Depends(require_roles(["admin"]))
):
    """
    Reset a circuit breaker.
    
    Manually resets the circuit breaker for a service.
    """
    healing = get_self_healing_service()
    
    try:
        await healing.reset_circuit_breaker(service)
        return {"message": f"Circuit breaker for '{service}' reset"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/recovery-actions")
async def get_recovery_actions(
    hours: int = Query(default=24, ge=1, le=168),
    current_user = Depends(require_roles(["admin"]))
):
    """
    Get recent recovery actions.
    
    Shows automatic recovery actions taken by the self-healing system.
    """
    healing = get_self_healing_service()
    actions = await healing.get_recent_actions(hours=hours)
    
    return {
        "total_actions": len(actions),
        "time_range_hours": hours,
        "actions": actions
    }


# ============================================================================
# Data Quality Endpoints
# ============================================================================

@router.get("/data-quality", response_model=DataQualityReport)
async def get_data_quality(
    db: AsyncSession = Depends(get_db),
    current_user = Depends(require_roles(["admin"]))
):
    """
    Get data quality report.
    
    Shows results of data quality checks and any issues found.
    """
    yesterday = datetime.utcnow() - timedelta(hours=24)
    
    query = select(DataQualityCheck).where(
        DataQualityCheck.checked_at >= yesterday
    ).order_by(DataQualityCheck.checked_at.desc())
    
    result = await db.execute(query)
    checks = result.scalars().all()
    
    passed = len([c for c in checks if c.passed])
    failed = len([c for c in checks if not c.passed])
    total = passed + failed
    score = (passed / total * 100) if total > 0 else 100.0
    
    issues = [
        {
            "check_name": c.check_name,
            "error": c.error_message,
            "checked_at": c.checked_at.isoformat()
        }
        for c in checks if not c.passed
    ]
    
    return DataQualityReport(
        overall_score=round(score, 2),
        checks_passed=passed,
        checks_failed=failed,
        last_check=checks[0].checked_at if checks else datetime.utcnow(),
        issues=issues[:20]
    )


@router.post("/data-quality/run")
async def run_data_quality_checks(
    current_user = Depends(require_roles(["admin"]))
):
    """
    Trigger data quality checks.
    
    Manually runs all data quality checks.
    """
    from app.services.data_quality import get_data_quality_service
    
    try:
        dq_service = get_data_quality_service()
        results = await dq_service.run_all_checks()
        return {
            "message": "Data quality checks completed",
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# System Info Endpoints
# ============================================================================

@router.get("/system-info")
async def get_system_info(
    current_user = Depends(require_roles(["admin"]))
):
    """
    Get system information.
    
    Returns server info including CPU, memory, disk, and GPU status.
    """
    import platform
    import psutil
    
    # GPU info if available
    gpu_info = None
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info = {
                "available": True,
                "device_count": torch.cuda.device_count(),
                "device_name": torch.cuda.get_device_name(0),
                "memory_allocated": torch.cuda.memory_allocated(0) / 1024**3,
                "memory_total": torch.cuda.get_device_properties(0).total_memory / 1024**3
            }
    except ImportError:
        gpu_info = {"available": False, "reason": "PyTorch not installed"}
    
    return {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "python_version": platform.python_version(),
        "cpu": {
            "physical_cores": psutil.cpu_count(logical=False),
            "total_cores": psutil.cpu_count(logical=True),
            "usage_percent": psutil.cpu_percent(interval=1)
        },
        "memory": {
            "total_gb": round(psutil.virtual_memory().total / 1024**3, 2),
            "available_gb": round(psutil.virtual_memory().available / 1024**3, 2),
            "usage_percent": psutil.virtual_memory().percent
        },
        "disk": {
            "total_gb": round(psutil.disk_usage('/').total / 1024**3, 2),
            "free_gb": round(psutil.disk_usage('/').free / 1024**3, 2),
            "usage_percent": psutil.disk_usage('/').percent
        },
        "gpu": gpu_info
    }


@router.get("/logs")
async def get_recent_logs(
    level: str = Query(default="INFO"),
    lines: int = Query(default=100, ge=10, le=1000),
    current_user = Depends(require_roles(["admin"]))
):
    """
    Get recent application logs.
    
    Returns recent log entries filtered by level.
    """
    import os
    
    log_file = os.getenv("LOG_FILE", "/var/log/loyaley/app.log")
    
    logs = []
    try:
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                all_lines = f.readlines()
                logs = all_lines[-lines:]
    except Exception as e:
        return {"error": str(e), "logs": []}
    
    # Filter by level
    level_order = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    min_level_idx = level_order.index(level.upper()) if level.upper() in level_order else 1
    
    filtered_logs = []
    for log in logs:
        for idx, lvl in enumerate(level_order):
            if lvl in log and idx >= min_level_idx:
                filtered_logs.append(log.strip())
                break
    
    return {
        "total_lines": len(filtered_logs),
        "level_filter": level,
        "logs": filtered_logs
    }
