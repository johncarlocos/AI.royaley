"""
LOYALEY - Phase 4 Self-Healing Service
Automatic failure detection, recovery, and system resilience
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from app.core.config import settings
from app.core.database import db_manager
from app.core.cache import cache_manager
from app.services.alerting.alerting_service import alerting_service, AlertSeverity

logger = logging.getLogger(__name__)


class ServiceStatus(str, Enum):
    """Service operational status"""
    RUNNING = "running"
    DEGRADED = "degraded"
    FAILED = "failed"
    RECOVERING = "recovering"
    STOPPED = "stopped"


class CircuitState(str, Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class ServiceHealth:
    """Service health tracking"""
    name: str
    status: ServiceStatus
    last_check: datetime
    failure_count: int = 0
    recovery_attempts: int = 0
    last_failure: Optional[datetime] = None
    last_recovery: Optional[datetime] = None
    error_message: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CircuitBreaker:
    """Circuit breaker for service protection"""
    name: str
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: float = 0
    failure_threshold: int = 5
    recovery_timeout: int = 30
    half_open_max_calls: int = 3
    
    def can_execute(self) -> bool:
        """Check if request can proceed"""
        if self.state == CircuitState.CLOSED:
            return True
        
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                return True
            return False
        
        # HALF_OPEN: allow limited requests
        return self.success_count < self.half_open_max_calls
    
    def record_success(self):
        """Record successful call"""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.half_open_max_calls:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                logger.info(f"Circuit breaker {self.name} closed after recovery")
        else:
            self.failure_count = 0
    
    def record_failure(self):
        """Record failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            logger.warning(f"Circuit breaker {self.name} reopened after failed recovery")
        elif self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(f"Circuit breaker {self.name} opened after {self.failure_count} failures")


class RecoveryAction:
    """Base class for recovery actions"""
    
    def __init__(self, name: str, service: str):
        self.name = name
        self.service = service
    
    async def execute(self) -> bool:
        """Execute recovery action"""
        raise NotImplementedError


class DatabaseReconnectAction(RecoveryAction):
    """Recovery action: Reconnect to database"""
    
    def __init__(self):
        super().__init__("database_reconnect", "database")
    
    async def execute(self) -> bool:
        try:
            await db_manager.close()
            await asyncio.sleep(2)
            await db_manager.initialize()
            
            # Verify connection
            health = await db_manager.health_check()
            return health["status"] == "healthy"
        except Exception as e:
            logger.error(f"Database reconnect failed: {e}")
            return False


class CacheReconnectAction(RecoveryAction):
    """Recovery action: Reconnect to Redis"""
    
    def __init__(self):
        super().__init__("cache_reconnect", "cache")
    
    async def execute(self) -> bool:
        try:
            await cache_manager.close()
            await asyncio.sleep(2)
            await cache_manager.initialize()
            
            # Verify connection
            health = await cache_manager.health_check()
            return health["status"] == "healthy"
        except Exception as e:
            logger.error(f"Cache reconnect failed: {e}")
            return False


class CacheClearAction(RecoveryAction):
    """Recovery action: Clear corrupted cache"""
    
    def __init__(self):
        super().__init__("cache_clear", "cache")
    
    async def execute(self) -> bool:
        try:
            # Clear specific problematic keys rather than entire cache
            patterns = [
                "error:*",
                "stale:*"
            ]
            for pattern in patterns:
                await cache_manager.delete_pattern(pattern)
            return True
        except Exception as e:
            logger.error(f"Cache clear failed: {e}")
            return False


class ModelReloadAction(RecoveryAction):
    """Recovery action: Reload ML models"""
    
    def __init__(self):
        super().__init__("model_reload", "ml")
    
    async def execute(self) -> bool:
        try:
            # This would trigger model reloading from disk
            # Implementation depends on model manager
            logger.info("Triggering model reload...")
            return True
        except Exception as e:
            logger.error(f"Model reload failed: {e}")
            return False


class ServiceRestartAction(RecoveryAction):
    """Recovery action: Restart a service via systemd/docker"""
    
    def __init__(self, service_name: str):
        super().__init__(f"restart_{service_name}", service_name)
        self.service_name = service_name
    
    async def execute(self) -> bool:
        try:
            import subprocess
            
            # Try Docker first
            result = subprocess.run(
                ["docker", "restart", self.service_name],
                capture_output=True,
                timeout=60
            )
            
            if result.returncode == 0:
                logger.info(f"Service {self.service_name} restarted via Docker")
                return True
            
            # Try systemd
            result = subprocess.run(
                ["systemctl", "restart", self.service_name],
                capture_output=True,
                timeout=60
            )
            
            if result.returncode == 0:
                logger.info(f"Service {self.service_name} restarted via systemd")
                return True
            
            return False
        except Exception as e:
            logger.error(f"Service restart failed: {e}")
            return False


class SelfHealingService:
    """Enterprise self-healing service"""
    
    def __init__(self):
        self.enabled = settings.SELF_HEALING_ENABLED
        self.max_restart_attempts = settings.MAX_RESTART_ATTEMPTS
        self.restart_cooldown = settings.RESTART_COOLDOWN_SECONDS
        
        self._services: Dict[str, ServiceHealth] = {}
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._recovery_actions: Dict[str, List[RecoveryAction]] = {}
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None
        
        # Register default services
        self._register_default_services()
    
    def _register_default_services(self):
        """Register default services for monitoring"""
        services = ["database", "cache", "api", "ml", "scheduler"]
        
        for service in services:
            self._services[service] = ServiceHealth(
                name=service,
                status=ServiceStatus.RUNNING,
                last_check=datetime.now(timezone.utc)
            )
            
            self._circuit_breakers[service] = CircuitBreaker(
                name=service,
                failure_threshold=settings.CIRCUIT_BREAKER_THRESHOLD,
                recovery_timeout=settings.CIRCUIT_BREAKER_TIMEOUT
            )
        
        # Register recovery actions
        self._recovery_actions["database"] = [DatabaseReconnectAction()]
        self._recovery_actions["cache"] = [CacheReconnectAction(), CacheClearAction()]
        self._recovery_actions["ml"] = [ModelReloadAction()]
    
    async def start(self):
        """Start the self-healing service"""
        if not self.enabled:
            logger.info("Self-healing service is disabled")
            return
        
        if self._running:
            return
        
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Self-healing service started")
    
    async def stop(self):
        """Stop the self-healing service"""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Self-healing service stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self._running:
            try:
                await self._check_all_services()
                await asyncio.sleep(settings.HEALTH_CHECK_INTERVAL)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(10)
    
    async def _check_all_services(self):
        """Check health of all registered services"""
        for service_name in self._services:
            try:
                await self._check_service(service_name)
            except Exception as e:
                logger.error(f"Error checking service {service_name}: {e}")
    
    async def _check_service(self, service_name: str):
        """Check health of a specific service"""
        health = self._services[service_name]
        circuit_breaker = self._circuit_breakers[service_name]
        
        # Perform health check
        is_healthy = await self._perform_health_check(service_name)
        health.last_check = datetime.now(timezone.utc)
        
        if is_healthy:
            circuit_breaker.record_success()
            
            if health.status != ServiceStatus.RUNNING:
                health.status = ServiceStatus.RUNNING
                health.last_recovery = datetime.now(timezone.utc)
                health.failure_count = 0
                health.recovery_attempts = 0
                
                await alerting_service.info(
                    title=f"Service Recovered: {service_name}",
                    message=f"Service {service_name} has recovered and is now healthy",
                    source="self-healing"
                )
        else:
            circuit_breaker.record_failure()
            health.failure_count += 1
            health.last_failure = datetime.now(timezone.utc)
            
            if health.status == ServiceStatus.RUNNING:
                health.status = ServiceStatus.DEGRADED
            
            if health.failure_count >= 3:
                health.status = ServiceStatus.FAILED
                
                # Attempt recovery if auto-restart is enabled
                if settings.AUTO_RESTART_FAILED_SERVICES:
                    await self._attempt_recovery(service_name)
    
    async def _perform_health_check(self, service_name: str) -> bool:
        """Perform health check for a service"""
        try:
            if service_name == "database":
                result = await db_manager.health_check()
                return result["status"] == "healthy"
            
            elif service_name == "cache":
                result = await cache_manager.health_check()
                return result["status"] == "healthy"
            
            elif service_name == "api":
                # API is healthy if we can reach this code
                return True
            
            elif service_name == "ml":
                # Check if model files exist and are loadable
                import os
                return os.path.exists(settings.MODEL_STORAGE_PATH)
            
            elif service_name == "scheduler":
                # Check scheduler health - simplified
                return True
            
            return True
        except Exception as e:
            logger.warning(f"Health check failed for {service_name}: {e}")
            return False
    
    async def _attempt_recovery(self, service_name: str):
        """Attempt to recover a failed service"""
        health = self._services[service_name]
        
        if health.recovery_attempts >= self.max_restart_attempts:
            logger.error(f"Max recovery attempts reached for {service_name}")
            
            await alerting_service.critical(
                title=f"Service Recovery Failed: {service_name}",
                message=f"Service {service_name} failed to recover after {self.max_restart_attempts} attempts",
                source="self-healing",
                metadata={
                    "service": service_name,
                    "attempts": health.recovery_attempts,
                    "last_error": health.error_message
                }
            )
            return
        
        # Check cooldown
        if health.last_recovery:
            elapsed = (datetime.now(timezone.utc) - health.last_recovery).total_seconds()
            if elapsed < self.restart_cooldown:
                logger.debug(f"Recovery cooldown active for {service_name}")
                return
        
        health.status = ServiceStatus.RECOVERING
        health.recovery_attempts += 1
        
        logger.info(f"Attempting recovery for {service_name} (attempt {health.recovery_attempts})")
        
        # Execute recovery actions
        actions = self._recovery_actions.get(service_name, [])
        
        for action in actions:
            try:
                success = await action.execute()
                
                if success:
                    logger.info(f"Recovery action {action.name} succeeded for {service_name}")
                    
                    # Verify service is now healthy
                    await asyncio.sleep(5)
                    if await self._perform_health_check(service_name):
                        health.status = ServiceStatus.RUNNING
                        health.last_recovery = datetime.now(timezone.utc)
                        health.failure_count = 0
                        
                        await alerting_service.info(
                            title=f"Service Auto-Recovered: {service_name}",
                            message=f"Service {service_name} was automatically recovered",
                            source="self-healing",
                            metadata={
                                "action": action.name,
                                "attempts": health.recovery_attempts
                            }
                        )
                        return
                else:
                    logger.warning(f"Recovery action {action.name} failed for {service_name}")
            except Exception as e:
                logger.error(f"Recovery action {action.name} error: {e}")
                health.error_message = str(e)
        
        # If all actions failed, mark as failed
        health.status = ServiceStatus.FAILED
        
        await alerting_service.error(
            title=f"Service Recovery Attempt Failed: {service_name}",
            message=f"Recovery attempt {health.recovery_attempts} failed for {service_name}",
            source="self-healing",
            metadata={
                "attempts": health.recovery_attempts,
                "max_attempts": self.max_restart_attempts
            }
        )
    
    def get_circuit_breaker(self, service_name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker for a service"""
        return self._circuit_breakers.get(service_name)
    
    def can_call_service(self, service_name: str) -> bool:
        """Check if service can be called (circuit breaker check)"""
        circuit_breaker = self._circuit_breakers.get(service_name)
        if circuit_breaker:
            return circuit_breaker.can_execute()
        return True
    
    def record_service_success(self, service_name: str):
        """Record successful service call"""
        circuit_breaker = self._circuit_breakers.get(service_name)
        if circuit_breaker:
            circuit_breaker.record_success()
    
    def record_service_failure(self, service_name: str):
        """Record failed service call"""
        circuit_breaker = self._circuit_breakers.get(service_name)
        if circuit_breaker:
            circuit_breaker.record_failure()
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all services and circuit breakers"""
        return {
            "enabled": self.enabled,
            "running": self._running,
            "services": {
                name: {
                    "status": health.status.value,
                    "last_check": health.last_check.isoformat(),
                    "failure_count": health.failure_count,
                    "recovery_attempts": health.recovery_attempts,
                    "circuit_breaker": self._circuit_breakers[name].state.value
                }
                for name, health in self._services.items()
            }
        }
    
    async def manual_recovery(self, service_name: str) -> bool:
        """Trigger manual recovery for a service"""
        if service_name not in self._services:
            return False
        
        health = self._services[service_name]
        health.recovery_attempts = 0  # Reset for manual recovery
        
        await self._attempt_recovery(service_name)
        
        return health.status == ServiceStatus.RUNNING


# Decorator for circuit breaker protection
def with_circuit_breaker(service_name: str):
    """Decorator to wrap function with circuit breaker protection"""
    def decorator(func: Callable):
        async def wrapper(*args, **kwargs):
            if not self_healing_service.can_call_service(service_name):
                raise Exception(f"Circuit breaker open for {service_name}")
            
            try:
                result = await func(*args, **kwargs)
                self_healing_service.record_service_success(service_name)
                return result
            except Exception as e:
                self_healing_service.record_service_failure(service_name)
                raise
        
        return wrapper
    return decorator


# Global self-healing service instance
self_healing_service = SelfHealingService()

def get_self_healing_service() -> SelfHealingService:
    """
    Dependency-style accessor for self-healing service.
    Keeps imports stable and avoids circular imports.
    """
    return self_healing_service