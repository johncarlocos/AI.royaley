"""
ROYALEY - Admin API Routes
System administration, user management, and configuration
"""

from datetime import datetime, timedelta
from typing import Optional, List
from enum import Enum

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from pydantic import BaseModel, Field, EmailStr
from sqlalchemy import select, func, delete, update, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_db
from app.core.database import get_database_manager
from app.api.dependencies import get_current_user, require_roles
from app.core.security import SecurityManager
from app.core.cache import CacheManager
from app.core.config import settings
from app.models import (
    User, Sport, Team, Game, Prediction, MLModel,
    AuditLog, Alert, ScheduledTask, DataQualityCheck
)

router = APIRouter()
security = SecurityManager()
cache = CacheManager()


# ============== Schemas ==============

class UserAdminResponse(BaseModel):
    """Admin user view"""
    id: int
    email: str
    username: str
    full_name: Optional[str]
    role: str
    is_active: bool
    two_factor_enabled: bool
    created_at: datetime
    last_login: Optional[datetime]
    predictions_count: Optional[int] = 0
    bets_count: Optional[int] = 0

    class Config:
        from_attributes = True


class UserUpdateRequest(BaseModel):
    """Admin user update"""
    role: Optional[str] = None
    is_active: Optional[bool] = None
    full_name: Optional[str] = None


class PasswordResetRequest(BaseModel):
    """Password reset request"""
    new_password: str = Field(..., min_length=8)


class SportConfig(BaseModel):
    """Sport configuration"""
    code: str
    name: str
    is_active: bool
    api_code: Optional[str]
    feature_count: int
    seasons: List[str]

    class Config:
        from_attributes = True


class SystemStats(BaseModel):
    """System statistics"""
    total_users: int
    active_users_24h: int
    total_predictions: int
    predictions_today: int
    total_games: int
    games_today: int
    total_bets: int
    models_in_production: int
    cache_hit_rate: float
    db_connections: int


class LogEntry(BaseModel):
    """System log entry"""
    id: int
    level: str
    component: str
    message: str
    details: Optional[dict]
    created_at: datetime

    class Config:
        from_attributes = True


class TaskStatus(BaseModel):
    """Scheduled task status"""
    id: int
    name: str
    schedule: str
    last_run: Optional[datetime]
    next_run: Optional[datetime]
    status: str
    enabled: bool

    class Config:
        from_attributes = True


class ConfigUpdate(BaseModel):
    """Configuration update"""
    key: str
    value: str
    description: Optional[str] = None


# ============== User Management ==============

@router.get("/users", response_model=List[UserAdminResponse])
async def list_users(
    role: Optional[str] = None,
    is_active: Optional[bool] = None,
    search: Optional[str] = None,
    page: int = 1,
    per_page: int = 50,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles(["admin"]))
):
    """
    List all users with filtering.
    
    **Admin only**
    """
    query = select(User)
    
    if role:
        query = query.where(User.role == role)
    if is_active is not None:
        query = query.where(User.is_active == is_active)
    if search:
        query = query.where(
            User.username.ilike(f"%{search}%") |
            User.email.ilike(f"%{search}%") |
            User.full_name.ilike(f"%{search}%")
        )
    
    query = query.order_by(User.created_at.desc())
    query = query.offset((page - 1) * per_page).limit(per_page)
    
    result = await db.execute(query)
    return result.scalars().all()


@router.get("/users/{user_id}", response_model=UserAdminResponse)
async def get_user(
    user_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles(["admin"]))
):
    """
    Get user details.
    
    **Admin only**
    """
    result = await db.execute(
        select(User).where(User.id == user_id)
    )
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return user


@router.patch("/users/{user_id}", response_model=UserAdminResponse)
async def update_user(
    user_id: int,
    update_data: UserUpdateRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles(["admin"]))
):
    """
    Update user settings.
    
    **Admin only**
    """
    if user_id == current_user.id and update_data.role:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot change your own role"
        )
    
    result = await db.execute(
        select(User).where(User.id == user_id)
    )
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    if update_data.role:
        user.role = update_data.role
    if update_data.is_active is not None:
        user.is_active = update_data.is_active
    if update_data.full_name is not None:
        user.full_name = update_data.full_name
    
    await db.commit()
    await db.refresh(user)
    
    return user


@router.delete("/users/{user_id}")
async def delete_user(
    user_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles(["admin"]))
):
    """
    Delete a user.
    
    **Admin only**
    """
    if user_id == current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete yourself"
        )
    
    result = await db.execute(
        select(User).where(User.id == user_id)
    )
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    await db.delete(user)
    await db.commit()
    
    return {"message": f"User {user_id} deleted"}


@router.post("/users/{user_id}/reset-password")
async def reset_user_password(
    user_id: int,
    request: PasswordResetRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles(["admin"]))
):
    """
    Reset a user's password.
    
    **Admin only**
    """
    result = await db.execute(
        select(User).where(User.id == user_id)
    )
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    user.password_hash = security.hash_password(request.new_password)
    user.two_factor_enabled = False
    user.two_factor_secret = None
    
    await db.commit()
    
    return {"message": "Password reset successfully"}


# ============== Sports Management ==============

@router.get("/sports", response_model=List[SportConfig])
async def list_sports(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles(["admin"]))
):
    """
    List all sports configurations.
    
    **Admin only**
    """
    result = await db.execute(select(Sport))
    return result.scalars().all()


@router.patch("/sports/{sport_code}")
async def update_sport(
    sport_code: str,
    is_active: Optional[bool] = None,
    api_code: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles(["admin"]))
):
    """
    Update sport configuration.
    
    **Admin only**
    """
    result = await db.execute(
        select(Sport).where(Sport.code == sport_code.upper())
    )
    sport = result.scalar_one_or_none()
    
    if not sport:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Sport not found"
        )
    
    if is_active is not None:
        sport.is_active = is_active
    if api_code is not None:
        sport.api_code = api_code
    
    await db.commit()
    
    return {"message": f"Sport {sport_code} updated"}


# ============== System Stats ==============

@router.get("/stats", response_model=SystemStats)
async def get_system_stats(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles(["admin"]))
):
    """
    Get system-wide statistics.
    
    **Admin only**
    """
    now = datetime.utcnow()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday = now - timedelta(hours=24)
    
    # User counts
    total_users = await db.execute(select(func.count(User.id)))
    active_24h = await db.execute(
        select(func.count(User.id)).where(User.last_login >= yesterday)
    )
    
    # Prediction counts
    total_preds = await db.execute(select(func.count(Prediction.id)))
    preds_today = await db.execute(
        select(func.count(Prediction.id)).where(Prediction.created_at >= today_start)
    )
    
    # Game counts
    total_games = await db.execute(select(func.count(Game.id)))
    games_today = await db.execute(
        select(func.count(Game.id)).where(Game.game_date >= today_start)
    )
    
    # Model count
    prod_models = await db.execute(
        select(func.count(MLModel.id)).where(MLModel.status == "production")
    )
    
    # Cache stats
    cache_stats = await cache.get_stats()
    
    # DB connections
    db_manager = get_database_manager()
    pool_status = db_manager.get_stats().get('pool', {}) if db_manager else "N/A"
    
    return SystemStats(
        total_users=total_users.scalar() or 0,
        active_users_24h=active_24h.scalar() or 0,
        total_predictions=total_preds.scalar() or 0,
        predictions_today=preds_today.scalar() or 0,
        total_games=total_games.scalar() or 0,
        games_today=games_today.scalar() or 0,
        total_bets=0,  # Would come from bets table
        models_in_production=prod_models.scalar() or 0,
        cache_hit_rate=cache_stats.get("hit_rate", 0),
        db_connections=0  # Would parse from pool_status
    )


# ============== Logs ==============

@router.get("/logs", response_model=List[LogEntry])
async def get_system_logs(
    level: Optional[str] = None,
    component: Optional[str] = None,
    since: Optional[datetime] = None,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles(["admin"]))
):
    """
    Get system logs.
    
    **Admin only**
    """
    query = select(AuditLog)
    
    if level:
        query = query.where(AuditLog.level == level.upper())
    if component:
        query = query.where(AuditLog.component == component)
    if since:
        query = query.where(AuditLog.created_at >= since)
    
    query = query.order_by(AuditLog.created_at.desc()).limit(limit)
    
    result = await db.execute(query)
    return result.scalars().all()


@router.delete("/logs")
async def clear_old_logs(
    days: int = 30,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles(["admin"]))
):
    """
    Clear logs older than specified days.
    
    **Admin only**
    """
    cutoff = datetime.utcnow() - timedelta(days=days)
    
    result = await db.execute(
        delete(AuditLog).where(AuditLog.created_at < cutoff)
    )
    await db.commit()
    
    return {"message": f"Deleted {result.rowcount} log entries"}


# ============== Alerts ==============

@router.get("/alerts")
async def get_alerts(
    acknowledged: Optional[bool] = None,
    severity: Optional[str] = None,
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles(["admin"]))
):
    """
    Get system alerts.
    
    **Admin only**
    """
    query = select(Alert)
    
    if acknowledged is not None:
        query = query.where(Alert.acknowledged == acknowledged)
    if severity:
        query = query.where(Alert.severity == severity)
    
    query = query.order_by(Alert.created_at.desc()).limit(limit)
    
    result = await db.execute(query)
    alerts = result.scalars().all()
    
    return [
        {
            "id": a.id,
            "type": a.type,
            "severity": a.severity,
            "message": a.message,
            "details": a.details,
            "acknowledged": a.acknowledged,
            "created_at": a.created_at
        }
        for a in alerts
    ]


@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles(["admin"]))
):
    """
    Acknowledge an alert.
    
    **Admin only**
    """
    result = await db.execute(
        select(Alert).where(Alert.id == alert_id)
    )
    alert = result.scalar_one_or_none()
    
    if not alert:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Alert not found"
        )
    
    alert.acknowledged = True
    alert.acknowledged_by = current_user.id
    alert.acknowledged_at = datetime.utcnow()
    
    await db.commit()
    
    return {"message": "Alert acknowledged"}


# ============== Scheduled Tasks ==============

@router.get("/tasks", response_model=List[TaskStatus])
async def list_scheduled_tasks(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles(["admin"]))
):
    """
    List scheduled tasks.
    
    **Admin only**
    """
    result = await db.execute(select(ScheduledTask))
    return result.scalars().all()


@router.post("/tasks/{task_id}/run")
async def run_task_now(
    task_id: int,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles(["admin"]))
):
    """
    Trigger a scheduled task to run immediately.
    
    **Admin only**
    """
    result = await db.execute(
        select(ScheduledTask).where(ScheduledTask.id == task_id)
    )
    task = result.scalar_one_or_none()
    
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found"
        )
    
    # Queue task execution
    background_tasks.add_task(execute_scheduled_task, task.name)
    
    return {"message": f"Task {task.name} queued for execution"}


async def execute_scheduled_task(task_name: str):
    """Execute a scheduled task"""
    # Implementation would call the appropriate service
    pass


@router.patch("/tasks/{task_id}")
async def update_task(
    task_id: int,
    enabled: Optional[bool] = None,
    schedule: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles(["admin"]))
):
    """
    Update task settings.
    
    **Admin only**
    """
    result = await db.execute(
        select(ScheduledTask).where(ScheduledTask.id == task_id)
    )
    task = result.scalar_one_or_none()
    
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found"
        )
    
    if enabled is not None:
        task.enabled = enabled
    if schedule is not None:
        task.schedule = schedule
    
    await db.commit()
    
    return {"message": f"Task {task_id} updated"}


# ============== Data Quality ==============

@router.get("/data-quality")
async def get_data_quality_status(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles(["admin"]))
):
    """
    Get data quality check results.
    
    **Admin only**
    """
    result = await db.execute(
        select(DataQualityCheck)
        .order_by(DataQualityCheck.checked_at.desc())
        .limit(50)
    )
    checks = result.scalars().all()
    
    return [
        {
            "id": c.id,
            "check_type": c.check_type,
            "target": c.target,
            "status": c.status,
            "score": c.score,
            "issues": c.issues,
            "checked_at": c.checked_at
        }
        for c in checks
    ]


@router.post("/data-quality/run")
async def run_data_quality_checks(
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles(["admin"]))
):
    """
    Trigger data quality checks.
    
    **Admin only**
    """
    background_tasks.add_task(execute_data_quality_checks)
    
    return {"message": "Data quality checks started"}


async def execute_data_quality_checks():
    """Run all data quality checks"""
    # Implementation in services/data_quality/data_quality_service.py
    pass


# ============== Cache Management ==============

@router.get("/cache/stats")
async def get_cache_stats(
    current_user: User = Depends(require_roles(["admin"]))
):
    """
    Get cache statistics.
    
    **Admin only**
    """
    stats = await cache.get_stats()
    return stats


@router.post("/cache/clear")
async def clear_cache(
    pattern: Optional[str] = None,
    current_user: User = Depends(require_roles(["admin"]))
):
    """
    Clear cache entries.
    
    **Admin only**
    
    - **pattern**: Optional pattern to match keys (e.g., "predictions:*")
    """
    if pattern:
        count = await cache.clear_pattern(pattern)
        return {"message": f"Cleared {count} cache entries matching {pattern}"}
    else:
        await cache.clear_all()
        return {"message": "All cache cleared"}


# ============== Database Management ==============

@router.get("/database/stats")
async def get_database_stats(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles(["admin"]))
):
    """
    Get database statistics.
    
    **Admin only**
    """
    # Table sizes
    tables = {}
    for table_name in ["users", "predictions", "games", "odds", "teams", "ml_models"]:
        try:
            result = await db.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
            tables[table_name] = result.scalar() or 0
        except:
            tables[table_name] = -1
    
    return {
        "tables": tables,
        "pool_size": settings.DATABASE_POOL_SIZE,
        "pool_status": get_database_manager().get_stats().get('pool', {})
    }


@router.post("/database/vacuum")
async def vacuum_database(
    table: Optional[str] = None,
    background_tasks: BackgroundTasks = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles(["admin"]))
):
    """
    Run VACUUM on database.
    
    **Admin only**
    """
    if table:
        await db.execute(text(f"VACUUM ANALYZE {table}"))
    else:
        await db.execute(text("VACUUM ANALYZE"))
    
    return {"message": f"VACUUM completed on {'all tables' if not table else table}"}


# ============== System Maintenance ==============

@router.post("/maintenance/cleanup")
async def run_cleanup(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_roles(["admin"]))
):
    """
    Run system cleanup tasks.
    
    **Admin only**
    
    - Clears old logs
    - Clears expired sessions
    - Clears old cache entries
    - Archives old predictions
    """
    background_tasks.add_task(execute_cleanup)
    
    return {"message": "Cleanup tasks started"}


async def execute_cleanup():
    """Execute cleanup tasks"""
    # Implementation would clean up old data
    pass


@router.post("/maintenance/backup")
async def create_backup(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_roles(["admin"]))
):
    """
    Create database backup.
    
    **Admin only**
    """
    background_tasks.add_task(execute_backup)
    
    return {"message": "Backup started"}


async def execute_backup():
    """Create database backup"""
    # Implementation would use pg_dump
    pass


@router.get("/config")
async def get_system_config(
    current_user: User = Depends(require_roles(["admin"]))
):
    """
    Get non-sensitive system configuration.
    
    **Admin only**
    """
    return {
        "app_name": settings.APP_NAME,
        "app_version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "debug": settings.DEBUG,
        "kelly_fraction": settings.KELLY_FRACTION,
        "max_bet_percent": settings.MAX_BET_PERCENT,
        "min_edge_threshold": settings.MIN_EDGE_THRESHOLD,
        "signal_tier_a_min": settings.SIGNAL_TIER_A_MIN,
        "signal_tier_b_min": settings.SIGNAL_TIER_B_MIN,
        "signal_tier_c_min": settings.SIGNAL_TIER_C_MIN,
        "cache_ttl_default": settings.CACHE_TTL_DEFAULT,
        "access_token_expire_minutes": settings.ACCESS_TOKEN_EXPIRE_MINUTES
    }
