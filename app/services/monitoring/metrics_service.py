"""
ROYALEY - Phase 4 Enterprise Monitoring Service
Comprehensive system monitoring with Prometheus metrics and health checks
"""

import asyncio
import logging
import os
import platform
import psutil
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Info,
    CollectorRegistry,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

from app.core.config import settings
from app.core.database import db_manager
from app.core.cache import cache_manager

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


class ComponentHealth:
    """Health status for a single component"""
    
    def __init__(
        self,
        name: str,
        status: HealthStatus,
        message: str = "",
        latency_ms: float = 0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.name = name
        self.status = status
        self.message = message
        self.latency_ms = latency_ms
        self.metadata = metadata or {}
        self.timestamp = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "latency_ms": round(self.latency_ms, 2),
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }


class MetricsRegistry:
    """Prometheus metrics registry for ROYALEY"""
    
    def __init__(self):
        self.registry = CollectorRegistry()
        
        # Application Info
        self.app_info = Info(
            'royaley_app',
            'Application information',
            registry=self.registry
        )
        self.app_info.info({
            'version': settings.APP_VERSION,
            'environment': settings.ENVIRONMENT,
            'python_version': platform.python_version()
        })
        
        # HTTP Request Metrics
        self.http_requests_total = Counter(
            'royaley_http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        
        self.http_request_duration = Histogram(
            'royaley_http_request_duration_seconds',
            'HTTP request duration in seconds',
            ['method', 'endpoint'],
            buckets=[.005, .01, .025, .05, .075, .1, .25, .5, .75, 1.0, 2.5, 5.0, 7.5, 10.0],
            registry=self.registry
        )
        
        # Prediction Metrics
        self.predictions_generated = Counter(
            'royaley_predictions_generated_total',
            'Total predictions generated',
            ['sport', 'bet_type', 'tier'],
            registry=self.registry
        )
        
        self.predictions_graded = Counter(
            'royaley_predictions_graded_total',
            'Total predictions graded',
            ['sport', 'result'],
            registry=self.registry
        )
        
        self.prediction_accuracy = Gauge(
            'royaley_prediction_accuracy',
            'Prediction accuracy by sport and tier',
            ['sport', 'tier'],
            registry=self.registry
        )
        
        self.prediction_edge = Gauge(
            'royaley_prediction_edge',
            'Average prediction edge',
            ['sport', 'tier'],
            registry=self.registry
        )
        
        # CLV Metrics
        self.clv_total = Gauge(
            'royaley_clv_total',
            'Total CLV by sport',
            ['sport'],
            registry=self.registry
        )
        
        self.clv_average = Gauge(
            'royaley_clv_average',
            'Average CLV by sport',
            ['sport'],
            registry=self.registry
        )
        
        # Model Metrics
        self.model_training_duration = Histogram(
            'royaley_model_training_duration_seconds',
            'Model training duration',
            ['sport', 'model_type'],
            buckets=[60, 300, 600, 1200, 1800, 3600, 7200],
            registry=self.registry
        )
        
        self.model_inference_duration = Histogram(
            'royaley_model_inference_duration_seconds',
            'Model inference duration',
            ['sport', 'model_type'],
            buckets=[.01, .025, .05, .1, .25, .5, 1.0],
            registry=self.registry
        )
        
        self.model_auc = Gauge(
            'royaley_model_auc',
            'Model AUC score',
            ['sport', 'model_type'],
            registry=self.registry
        )
        
        # Betting Metrics
        self.bets_placed = Counter(
            'royaley_bets_placed_total',
            'Total bets placed',
            ['sport', 'bet_type'],
            registry=self.registry
        )
        
        self.bankroll_value = Gauge(
            'royaley_bankroll_value',
            'Current bankroll value',
            ['user_id'],
            registry=self.registry
        )
        
        self.roi = Gauge(
            'royaley_roi',
            'Return on investment',
            ['sport', 'period'],
            registry=self.registry
        )
        
        # Data Collection Metrics
        self.data_collection_runs = Counter(
            'royaley_data_collection_runs_total',
            'Total data collection runs',
            ['source', 'status'],
            registry=self.registry
        )
        
        self.odds_collected = Counter(
            'royaley_odds_collected_total',
            'Total odds records collected',
            ['sport', 'sportsbook'],
            registry=self.registry
        )
        
        self.api_rate_limit_remaining = Gauge(
            'royaley_api_rate_limit_remaining',
            'API rate limit remaining',
            ['api'],
            registry=self.registry
        )
        
        # System Metrics
        self.cpu_usage = Gauge(
            'royaley_cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )
        
        self.memory_usage = Gauge(
            'royaley_memory_usage_bytes',
            'Memory usage in bytes',
            ['type'],
            registry=self.registry
        )
        
        self.disk_usage = Gauge(
            'royaley_disk_usage_bytes',
            'Disk usage in bytes',
            ['mount', 'type'],
            registry=self.registry
        )
        
        self.gpu_usage = Gauge(
            'royaley_gpu_usage_percent',
            'GPU usage percentage',
            ['gpu_id'],
            registry=self.registry
        )
        
        self.gpu_memory = Gauge(
            'royaley_gpu_memory_bytes',
            'GPU memory in bytes',
            ['gpu_id', 'type'],
            registry=self.registry
        )
        
        # Database Metrics
        self.db_connections = Gauge(
            'royaley_db_connections',
            'Database connections',
            ['state'],
            registry=self.registry
        )
        
        self.db_query_duration = Histogram(
            'royaley_db_query_duration_seconds',
            'Database query duration',
            ['operation'],
            buckets=[.001, .005, .01, .025, .05, .1, .25, .5, 1.0],
            registry=self.registry
        )
        
        # Cache Metrics
        self.cache_hits = Counter(
            'royaley_cache_hits_total',
            'Total cache hits',
            registry=self.registry
        )
        
        self.cache_misses = Counter(
            'royaley_cache_misses_total',
            'Total cache misses',
            registry=self.registry
        )
        
        # Error Metrics
        self.errors = Counter(
            'royaley_errors_total',
            'Total errors',
            ['type', 'component'],
            registry=self.registry
        )
        
        # Alert Metrics
        self.alerts_sent = Counter(
            'royaley_alerts_sent_total',
            'Total alerts sent',
            ['channel', 'severity'],
            registry=self.registry
        )
    
    def get_metrics(self) -> bytes:
        """Get Prometheus metrics output"""
        return generate_latest(self.registry)
    
    def get_content_type(self) -> str:
        """Get Prometheus content type"""
        return CONTENT_TYPE_LATEST


class MonitoringService:
    """Enterprise monitoring service"""
    
    def __init__(self):
        self.metrics = MetricsRegistry()
        self._health_checks: Dict[str, Callable] = {}
        self._last_health_check: Optional[Dict[str, Any]] = None
        self._start_time = time.time()
    
    def register_health_check(self, name: str, check_func: Callable):
        """Register a health check function"""
        self._health_checks[name] = check_func
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Update Prometheus gauges
        self.metrics.cpu_usage.set(cpu_percent)
        self.metrics.memory_usage.labels(type='used').set(memory.used)
        self.metrics.memory_usage.labels(type='available').set(memory.available)
        self.metrics.disk_usage.labels(mount='/', type='used').set(disk.used)
        self.metrics.disk_usage.labels(mount='/', type='free').set(disk.free)
        
        # Try to get GPU metrics
        gpu_metrics = await self._get_gpu_metrics()
        
        return {
            "cpu": {
                "percent": cpu_percent,
                "count": psutil.cpu_count()
            },
            "memory": {
                "total": memory.total,
                "used": memory.used,
                "available": memory.available,
                "percent": memory.percent
            },
            "disk": {
                "total": disk.total,
                "used": disk.used,
                "free": disk.free,
                "percent": disk.percent
            },
            "gpu": gpu_metrics,
            "uptime_seconds": time.time() - self._start_time
        }
    
    async def _get_gpu_metrics(self) -> Optional[Dict[str, Any]]:
        """Get GPU metrics if available"""
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                gpus = []
                for i, line in enumerate(lines):
                    parts = line.split(',')
                    if len(parts) >= 3:
                        utilization = float(parts[0].strip())
                        memory_used = float(parts[1].strip()) * 1024 * 1024  # MB to bytes
                        memory_total = float(parts[2].strip()) * 1024 * 1024
                        
                        self.metrics.gpu_usage.labels(gpu_id=str(i)).set(utilization)
                        self.metrics.gpu_memory.labels(gpu_id=str(i), type='used').set(memory_used)
                        self.metrics.gpu_memory.labels(gpu_id=str(i), type='total').set(memory_total)
                        
                        gpus.append({
                            "id": i,
                            "utilization_percent": utilization,
                            "memory_used_mb": memory_used / (1024 * 1024),
                            "memory_total_mb": memory_total / (1024 * 1024)
                        })
                return {"available": True, "gpus": gpus}
        except Exception:
            pass
        return {"available": False}
    
    async def check_health(self, detailed: bool = False) -> Dict[str, Any]:
        """
        Perform health check on all components
        Returns overall health status and component details
        """
        components: List[ComponentHealth] = []
        
        # Database health
        db_health = await self._check_database_health()
        components.append(db_health)
        
        # Redis health
        redis_health = await self._check_redis_health()
        components.append(redis_health)
        
        # Run custom health checks
        for name, check_func in self._health_checks.items():
            try:
                start = time.time()
                result = await check_func() if asyncio.iscoroutinefunction(check_func) else check_func()
                latency = (time.time() - start) * 1000
                
                status = HealthStatus.HEALTHY if result.get("healthy", False) else HealthStatus.UNHEALTHY
                components.append(ComponentHealth(
                    name=name,
                    status=status,
                    message=result.get("message", ""),
                    latency_ms=latency,
                    metadata=result.get("metadata", {})
                ))
            except Exception as e:
                components.append(ComponentHealth(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=str(e)
                ))
        
        # Calculate overall status
        overall_status = self._calculate_overall_status(components)
        
        result = {
            "status": overall_status.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": settings.APP_VERSION,
            "environment": settings.ENVIRONMENT,
            "uptime_seconds": int(time.time() - self._start_time)
        }
        
        if detailed:
            result["components"] = [c.to_dict() for c in components]
            result["system"] = await self.get_system_metrics()
        
        self._last_health_check = result
        return result
    
    async def _check_database_health(self) -> ComponentHealth:
        """Check database health"""
        try:
            start = time.time()
            health = await db_manager.health_check()
            latency = (time.time() - start) * 1000
            
            status = HealthStatus.HEALTHY if health["status"] == "healthy" else HealthStatus.UNHEALTHY
            
            # Update metrics
            if "pool" in health:
                pool = health["pool"]
                self.metrics.db_connections.labels(state='active').set(pool.get('checked_out', 0))
                self.metrics.db_connections.labels(state='idle').set(pool.get('checked_in', 0))
            
            return ComponentHealth(
                name="database",
                status=status,
                message="Database connection healthy" if status == HealthStatus.HEALTHY else health.get("error", ""),
                latency_ms=latency,
                metadata=health
            )
        except Exception as e:
            return ComponentHealth(
                name="database",
                status=HealthStatus.CRITICAL,
                message=str(e)
            )
    
    async def _check_redis_health(self) -> ComponentHealth:
        """Check Redis health"""
        try:
            start = time.time()
            health = await cache_manager.health_check()
            latency = (time.time() - start) * 1000
            
            status = HealthStatus.HEALTHY if health["status"] == "healthy" else HealthStatus.UNHEALTHY
            
            # Update metrics from cache stats
            stats = health.get("stats", {})
            if stats.get("hits"):
                self.metrics.cache_hits._value.set(stats["hits"])
            if stats.get("misses"):
                self.metrics.cache_misses._value.set(stats["misses"])
            
            return ComponentHealth(
                name="redis",
                status=status,
                message="Redis connection healthy" if status == HealthStatus.HEALTHY else health.get("error", ""),
                latency_ms=latency,
                metadata=health
            )
        except Exception as e:
            return ComponentHealth(
                name="redis",
                status=HealthStatus.UNHEALTHY,
                message=str(e)
            )
    
    def _calculate_overall_status(self, components: List[ComponentHealth]) -> HealthStatus:
        """Calculate overall health status from components"""
        statuses = [c.status for c in components]
        
        if HealthStatus.CRITICAL in statuses:
            return HealthStatus.CRITICAL
        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        if HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        return HealthStatus.HEALTHY
    
    def record_http_request(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        duration: float
    ):
        """Record HTTP request metrics"""
        self.metrics.http_requests_total.labels(
            method=method,
            endpoint=endpoint,
            status=str(status_code)
        ).inc()
        
        self.metrics.http_request_duration.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
    
    def record_prediction(
        self,
        sport: str,
        bet_type: str,
        tier: str
    ):
        """Record prediction generation"""
        self.metrics.predictions_generated.labels(
            sport=sport,
            bet_type=bet_type,
            tier=tier
        ).inc()
    
    def record_prediction_graded(
        self,
        sport: str,
        result: str
    ):
        """Record prediction grading"""
        self.metrics.predictions_graded.labels(
            sport=sport,
            result=result
        ).inc()
    
    def update_prediction_accuracy(
        self,
        sport: str,
        tier: str,
        accuracy: float
    ):
        """Update prediction accuracy gauge"""
        self.metrics.prediction_accuracy.labels(
            sport=sport,
            tier=tier
        ).set(accuracy)
    
    def record_error(
        self,
        error_type: str,
        component: str
    ):
        """Record error occurrence"""
        self.metrics.errors.labels(
            type=error_type,
            component=component
        ).inc()
    
    def record_model_training(
        self,
        sport: str,
        model_type: str,
        duration_seconds: float,
        auc: float
    ):
        """Record model training metrics"""
        self.metrics.model_training_duration.labels(
            sport=sport,
            model_type=model_type
        ).observe(duration_seconds)
        
        self.metrics.model_auc.labels(
            sport=sport,
            model_type=model_type
        ).set(auc)
    
    def get_prometheus_metrics(self) -> bytes:
        """Get Prometheus metrics for scraping"""
        return self.metrics.get_metrics()
    
    def get_prometheus_content_type(self) -> str:
        """Get Prometheus content type"""
        return self.metrics.get_content_type()


# Global monitoring service instance
monitoring_service = MonitoringService()

def get_monitoring_service() -> MonitoringService:
    """
    Dependency-style accessor for monitoring service.
    Keeps imports stable and avoids circular imports.
    """
    return monitoring_service
