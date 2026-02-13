"""
ROYALEY - Phase 4 Enterprise Scheduling Service
Background task scheduling with APScheduler
"""

import asyncio
import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.executors.asyncio import AsyncIOExecutor
from apscheduler.events import (
    EVENT_JOB_EXECUTED,
    EVENT_JOB_ERROR,
    EVENT_JOB_MISSED,
    JobExecutionEvent,
)

from app.core.config import settings
from app.services.alerting.alerting_service import alerting_service, AlertSeverity

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """Job execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    MISSED = "missed"


class JobCategory(str, Enum):
    """Job categories"""
    DATA_COLLECTION = "data_collection"
    PREDICTIONS = "predictions"
    GRADING = "grading"
    ML_TRAINING = "ml_training"
    MAINTENANCE = "maintenance"
    REPORTING = "reporting"


class ScheduledJob:
    """Scheduled job definition"""
    
    def __init__(
        self,
        job_id: str,
        name: str,
        category: JobCategory,
        func: Callable,
        trigger: str,  # 'interval' or 'cron'
        trigger_args: Dict[str, Any],
        enabled: bool = True,
        max_instances: int = 1,
        coalesce: bool = True,
        misfire_grace_time: int = 60
    ):
        self.job_id = job_id
        self.name = name
        self.category = category
        self.func = func
        self.trigger = trigger
        self.trigger_args = trigger_args
        self.enabled = enabled
        self.max_instances = max_instances
        self.coalesce = coalesce
        self.misfire_grace_time = misfire_grace_time
        
        # Execution tracking
        self.last_run: Optional[datetime] = None
        self.last_status: JobStatus = JobStatus.PENDING
        self.run_count: int = 0
        self.error_count: int = 0
        self.last_error: Optional[str] = None
        self.avg_duration_ms: float = 0


class SchedulerService:
    """Enterprise scheduling service"""
    
    def __init__(self):
        self.enabled = settings.SCHEDULER_ENABLED
        self._scheduler: Optional[AsyncIOScheduler] = None
        self._jobs: Dict[str, ScheduledJob] = {}
        self._running = False
    
    async def initialize(self):
        """Initialize the scheduler"""
        if not self.enabled:
            logger.info("Scheduler is disabled")
            return
        
        # Configure job stores and executors
        jobstores = {
            'default': MemoryJobStore()
        }
        
        executors = {
            'default': AsyncIOExecutor()
        }
        
        job_defaults = {
            'coalesce': True,
            'max_instances': 3,
            'misfire_grace_time': 60
        }
        
        self._scheduler = AsyncIOScheduler(
            jobstores=jobstores,
            executors=executors,
            job_defaults=job_defaults,
            timezone='UTC'
        )
        
        # Add event listeners
        self._scheduler.add_listener(self._on_job_executed, EVENT_JOB_EXECUTED)
        self._scheduler.add_listener(self._on_job_error, EVENT_JOB_ERROR)
        self._scheduler.add_listener(self._on_job_missed, EVENT_JOB_MISSED)
        
        # Register default jobs
        await self._register_default_jobs()
        
        logger.info("Scheduler initialized")
    
    async def start(self):
        """Start the scheduler"""
        if not self.enabled or not self._scheduler:
            return
        
        if self._running:
            return
        
        self._scheduler.start()
        self._running = True
        
        # Run initial collections immediately on startup
        asyncio.create_task(self._run_initial_odds_collection())
        asyncio.create_task(self._run_initial_player_props_collection())
        asyncio.create_task(self._run_initial_scores_collection())
        
        logger.info("Scheduler started")
    
    async def stop(self):
        """Stop the scheduler"""
        if self._scheduler and self._running:
            self._scheduler.shutdown(wait=True)
            self._running = False
            logger.info("Scheduler stopped")
    
    async def _register_default_jobs(self):
        """Register default scheduled jobs"""
        default_jobs = [
            # Data Collection Jobs
            ScheduledJob(
                job_id="collect_odds",
                name="Collect Live Odds",
                category=JobCategory.DATA_COLLECTION,
                func=self._collect_odds_job,  # Actual implementation
                trigger="interval",
                trigger_args={"seconds": settings.ODDS_REFRESH_INTERVAL}
            ),
            ScheduledJob(
                job_id="collect_games",
                name="Collect Game Schedules",
                category=JobCategory.DATA_COLLECTION,
                func=self._dummy_job,
                trigger="interval",
                trigger_args={"seconds": settings.GAMES_REFRESH_INTERVAL}
            ),
            ScheduledJob(
                job_id="collect_results",
                name="Collect Game Results",
                category=JobCategory.DATA_COLLECTION,
                func=self._dummy_job,
                trigger="interval",
                trigger_args={"seconds": 900}
            ),
            
            # Prediction Jobs
            ScheduledJob(
                job_id="generate_predictions",
                name="Generate Predictions",
                category=JobCategory.PREDICTIONS,
                func=self._dummy_job,
                trigger="interval",
                trigger_args={"seconds": settings.PREDICTION_GENERATION_INTERVAL}
            ),
            
            # Grading Jobs
            ScheduledJob(
                job_id="grade_predictions",
                name="Grade Predictions",
                category=JobCategory.GRADING,
                func=self._dummy_job,
                trigger="interval",
                trigger_args={"seconds": settings.GRADING_INTERVAL}
            ),
            ScheduledJob(
                job_id="calculate_clv",
                name="Calculate CLV",
                category=JobCategory.GRADING,
                func=self._dummy_job,
                trigger="interval",
                trigger_args={"seconds": 1800}
            ),
            
            # ML Training Jobs
            ScheduledJob(
                job_id="model_retraining",
                name="Model Retraining",
                category=JobCategory.ML_TRAINING,
                func=self._dummy_job,
                trigger="cron",
                trigger_args=self._parse_cron(settings.MODEL_RETRAINING_CRON)
            ),
            ScheduledJob(
                job_id="model_evaluation",
                name="Model Evaluation",
                category=JobCategory.ML_TRAINING,
                func=self._dummy_job,
                trigger="cron",
                trigger_args={"hour": 6, "minute": 0}
            ),
            
            # Maintenance Jobs
            ScheduledJob(
                job_id="data_cleanup",
                name="Data Cleanup",
                category=JobCategory.MAINTENANCE,
                func=self._dummy_job,
                trigger="cron",
                trigger_args=self._parse_cron(settings.DATA_CLEANUP_CRON)
            ),
            ScheduledJob(
                job_id="cache_cleanup",
                name="Cache Cleanup",
                category=JobCategory.MAINTENANCE,
                func=self._dummy_job,
                trigger="cron",
                trigger_args={"hour": 3, "minute": 0}
            ),
            ScheduledJob(
                job_id="health_check",
                name="System Health Check",
                category=JobCategory.MAINTENANCE,
                func=self._dummy_job,
                trigger="interval",
                trigger_args={"seconds": settings.HEALTH_CHECK_INTERVAL}
            ),
            
            # Reporting Jobs
            ScheduledJob(
                job_id="daily_report",
                name="Daily Performance Report",
                category=JobCategory.REPORTING,
                func=self._dummy_job,
                trigger="cron",
                trigger_args={"hour": 7, "minute": 0}
            ),
            ScheduledJob(
                job_id="weekly_report",
                name="Weekly Performance Report",
                category=JobCategory.REPORTING,
                func=self._dummy_job,
                trigger="cron",
                trigger_args={"day_of_week": "mon", "hour": 8, "minute": 0}
            ),
            
            # Player Props Pipeline
            ScheduledJob(
                job_id="collect_player_props",
                name="Collect Player Props",
                category=JobCategory.DATA_COLLECTION,
                func=self._collect_player_props_job,
                trigger="interval",
                trigger_args={"seconds": 14400}  # Every 4 hours
            ),
            
            # Live Scores Pipeline
            ScheduledJob(
                job_id="update_live_scores",
                name="Update Live Scores",
                category=JobCategory.DATA_COLLECTION,
                func=self._update_live_scores_job,
                trigger="interval",
                trigger_args={"seconds": 120}  # Every 2 minutes
            ),
        ]
        
        for job in default_jobs:
            self.register_job(job)
    
    def _parse_cron(self, cron_str: str) -> Dict[str, Any]:
        """Parse cron string to trigger args"""
        parts = cron_str.split()
        if len(parts) != 5:
            return {"hour": 4, "minute": 0}  # Default
        
        return {
            "minute": parts[0] if parts[0] != "*" else None,
            "hour": parts[1] if parts[1] != "*" else None,
            "day": parts[2] if parts[2] != "*" else None,
            "month": parts[3] if parts[3] != "*" else None,
            "day_of_week": parts[4] if parts[4] != "*" else None,
        }
    
    async def _dummy_job(self):
        """Placeholder job function"""
        pass
    
    async def _collect_odds_job(self):
        """Collect odds from TheOddsAPI and save to database."""
        try:
            from app.services.collectors.collector_02_odds_api import odds_collector
            from app.core.database import db_manager
            from app.core.cache import cache_manager
            
            logger.info("[Scheduler] Starting odds collection job...")
            
            # Collect odds from TheOddsAPI
            result = await odds_collector.collect()
            
            if not result.success:
                logger.error(f"[Scheduler] Odds collection failed: {result.error}")
                return
            
            # Save to database
            saved_count = 0
            if result.data:
                async with db_manager.session() as session:
                    try:
                        saved_count = await odds_collector.save_to_database(result.data, session)
                        await session.commit()
                        logger.info(f"[Scheduler] ✅ Saved {saved_count} odds records to database")
                    except Exception as e:
                        await session.rollback()
                        logger.error(f"[Scheduler] Error saving odds to database: {e}")
                        raise
            
            # Clear odds cache
            await cache_manager.delete_pattern("odds:*")
            
            logger.info(f"[Scheduler] ✅ Odds collection completed: {result.records_count} collected, {saved_count} saved")
            
        except Exception as e:
            logger.error(f"[Scheduler] Error in odds collection job: {e}", exc_info=True)
            # Don't raise - let scheduler continue with other jobs
    
    async def _run_initial_odds_collection(self):
        """Run odds collection immediately on startup."""
        # Small delay to ensure all services are ready
        await asyncio.sleep(5)
        try:
            logger.info("[Scheduler] Running initial odds collection on startup...")
            await self._collect_odds_job()
        except Exception as e:
            logger.error(f"[Scheduler] Initial odds collection failed: {e}", exc_info=True)
    
    async def _collect_player_props_job(self):
        """Collect player prop lines from The Odds API and generate predictions."""
        try:
            from app.pipeline.fetch_player_props import run_player_props_pipeline
            
            logger.info("[Scheduler] Starting player props collection...")
            await run_player_props_pipeline(max_events_per_sport=10)
            logger.info("[Scheduler] ✅ Player props collection completed")
            
        except Exception as e:
            logger.error(f"[Scheduler] Error in player props job: {e}", exc_info=True)
    
    async def _run_initial_player_props_collection(self):
        """Run player props collection on startup after a delay."""
        await asyncio.sleep(15)  # Wait for DB and other services
        try:
            logger.info("[Scheduler] Running initial player props collection on startup...")
            await self._collect_player_props_job()
        except Exception as e:
            logger.error(f"[Scheduler] Initial player props collection failed: {e}", exc_info=True)
    
    async def _update_live_scores_job(self):
        """Fetch live scores from The Odds API and update upcoming_games."""
        try:
            from app.pipeline.fetch_scores import run_scores_pipeline
            
            logger.info("[Scheduler] Updating live scores...")
            await run_scores_pipeline(days_from=1)
            logger.info("[Scheduler] ✅ Live scores updated")
            
        except Exception as e:
            logger.error(f"[Scheduler] Error in live scores job: {e}", exc_info=True)
    
    async def _run_initial_scores_collection(self):
        """Run score collection on startup after a delay."""
        await asyncio.sleep(20)
        try:
            logger.info("[Scheduler] Running initial live scores collection on startup...")
            await self._update_live_scores_job()
        except Exception as e:
            logger.error(f"[Scheduler] Initial scores collection failed: {e}", exc_info=True)
    
    def register_job(self, job: ScheduledJob):
        """Register a scheduled job"""
        if not self._scheduler:
            logger.warning("Scheduler not initialized, cannot register job")
            return
        
        self._jobs[job.job_id] = job
        
        if not job.enabled:
            return
        
        # Create trigger
        if job.trigger == "interval":
            trigger = IntervalTrigger(**job.trigger_args)
        else:
            trigger = CronTrigger(**{k: v for k, v in job.trigger_args.items() if v is not None})
        
        # Add job to scheduler
        self._scheduler.add_job(
            job.func,
            trigger=trigger,
            id=job.job_id,
            name=job.name,
            max_instances=job.max_instances,
            coalesce=job.coalesce,
            misfire_grace_time=job.misfire_grace_time,
            replace_existing=True
        )
        
        logger.info(f"Registered job: {job.job_id} ({job.name})")
    
    def update_job_function(self, job_id: str, func: Callable):
        """Update the function for a registered job"""
        if job_id not in self._jobs:
            return False
        
        job = self._jobs[job_id]
        job.func = func
        
        # Re-register with new function
        self.register_job(job)
        return True
    
    def enable_job(self, job_id: str):
        """Enable a job"""
        if job_id not in self._jobs:
            return False
        
        job = self._jobs[job_id]
        job.enabled = True
        self.register_job(job)
        return True
    
    def disable_job(self, job_id: str):
        """Disable a job"""
        if job_id not in self._jobs:
            return False
        
        job = self._jobs[job_id]
        job.enabled = False
        
        if self._scheduler:
            try:
                self._scheduler.remove_job(job_id)
            except Exception:
                pass
        
        return True
    
    def run_job_now(self, job_id: str):
        """Trigger immediate job execution"""
        if not self._scheduler or job_id not in self._jobs:
            return False
        
        try:
            job = self._scheduler.get_job(job_id)
            if job:
                job.modify(next_run_time=datetime.now(timezone.utc))
                return True
        except Exception as e:
            logger.error(f"Error running job {job_id}: {e}")
        
        return False
    
    def _on_job_executed(self, event: JobExecutionEvent):
        """Handler for successful job execution"""
        job_id = event.job_id
        
        if job_id in self._jobs:
            job = self._jobs[job_id]
            job.last_run = datetime.now(timezone.utc)
            job.last_status = JobStatus.COMPLETED
            job.run_count += 1
            
            # Update average duration
            if hasattr(event, 'retval'):
                duration = event.scheduled_run_time
                # Simplified duration tracking
        
        logger.debug(f"Job executed: {job_id}")
    
    def _on_job_error(self, event: JobExecutionEvent):
        """Handler for job execution error"""
        job_id = event.job_id
        
        if job_id in self._jobs:
            job = self._jobs[job_id]
            job.last_run = datetime.now(timezone.utc)
            job.last_status = JobStatus.FAILED
            job.error_count += 1
            job.last_error = str(event.exception) if event.exception else "Unknown error"
        
        logger.error(f"Job failed: {job_id} - {event.exception}")
        
        # Send alert for failed jobs
        asyncio.create_task(
            alerting_service.error(
                title=f"Scheduled Job Failed: {job_id}",
                message=f"Job execution failed with error: {event.exception}",
                source="scheduler",
                metadata={"job_id": job_id}
            )
        )
    
    def _on_job_missed(self, event: JobExecutionEvent):
        """Handler for missed job execution"""
        job_id = event.job_id
        
        if job_id in self._jobs:
            job = self._jobs[job_id]
            job.last_status = JobStatus.MISSED
        
        logger.warning(f"Job missed: {job_id}")
        
        # Send alert for missed jobs
        asyncio.create_task(
            alerting_service.warning(
                title=f"Scheduled Job Missed: {job_id}",
                message=f"Job execution was missed",
                source="scheduler",
                metadata={"job_id": job_id}
            )
        )
    
    def get_jobs(self, category: Optional[JobCategory] = None) -> List[Dict[str, Any]]:
        """Get list of registered jobs"""
        jobs = []
        
        for job_id, job in self._jobs.items():
            if category and job.category != category:
                continue
            
            next_run = None
            if self._scheduler:
                scheduler_job = self._scheduler.get_job(job_id)
                if scheduler_job:
                    next_run = scheduler_job.next_run_time
            
            jobs.append({
                "job_id": job.job_id,
                "name": job.name,
                "category": job.category.value,
                "enabled": job.enabled,
                "trigger": job.trigger,
                "trigger_args": job.trigger_args,
                "last_run": job.last_run.isoformat() if job.last_run else None,
                "last_status": job.last_status.value,
                "next_run": next_run.isoformat() if next_run else None,
                "run_count": job.run_count,
                "error_count": job.error_count,
                "last_error": job.last_error
            })
        
        return jobs
    
    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get details of a specific job"""
        if job_id not in self._jobs:
            return None
        
        job = self._jobs[job_id]
        
        next_run = None
        if self._scheduler:
            scheduler_job = self._scheduler.get_job(job_id)
            if scheduler_job:
                next_run = scheduler_job.next_run_time
        
        return {
            "job_id": job.job_id,
            "name": job.name,
            "category": job.category.value,
            "enabled": job.enabled,
            "trigger": job.trigger,
            "trigger_args": job.trigger_args,
            "last_run": job.last_run.isoformat() if job.last_run else None,
            "last_status": job.last_status.value,
            "next_run": next_run.isoformat() if next_run else None,
            "run_count": job.run_count,
            "error_count": job.error_count,
            "last_error": job.last_error,
            "max_instances": job.max_instances,
            "avg_duration_ms": job.avg_duration_ms
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get scheduler status"""
        return {
            "enabled": self.enabled,
            "running": self._running,
            "total_jobs": len(self._jobs),
            "enabled_jobs": sum(1 for j in self._jobs.values() if j.enabled),
            "jobs_by_category": {
                cat.value: sum(1 for j in self._jobs.values() if j.category == cat)
                for cat in JobCategory
            }
        }


# Global scheduler service instance
scheduler_service = SchedulerService()

def get_scheduler_service() -> SchedulerService:
    """
    Dependency-style accessor for scheduler service.
    Keeps imports stable and avoids circular imports.
    """
    return scheduler_service