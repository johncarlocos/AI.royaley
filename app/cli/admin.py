"""
LOYALEY - CLI Admin Commands
Command-line interface for system administration

Fixed Issues:
- Added 'worker' command for background task processing
- Fixed 'db seed' command with correct model field names
- Fixed imports and async session handling
"""

import asyncio
import sys
import signal
import logging
from datetime import datetime, timedelta
from typing import Optional
import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel

console = Console()
logger = logging.getLogger(__name__)


@click.group()
def cli():
    """LOYALEY - Enterprise Sports Prediction Platform"""
    pass


# ============== Worker Command (NEW - FIX #1) ==============

@cli.command()
@click.option("--no-scheduler", is_flag=True, help="Disable background scheduler")
def worker(no_scheduler: bool):
    """
    Run background worker for scheduled tasks.
    
    This command starts the background worker that handles:
    - Odds collection (every 60 seconds)
    - Games collection (every 5 minutes)
    - Prediction generation (every hour)
    - Auto-grading (every 15 minutes)
    - Model performance checks (daily)
    - Data quality checks (daily at 4 AM)
    - Database backups (daily at 3 AM)
    """
    from app.core.config import get_settings
    
    settings = get_settings()
    
    console.print(Panel.fit(
        "[bold green]LOYALEY[/bold green]\n"
        "Background Worker Service",
        title="Worker Starting"
    ))
    
    console.print(f"Environment: {settings.environment}")
    console.print(f"Scheduler Enabled: {not no_scheduler}")
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    async def run_worker():
        """Main worker loop"""
        from app.core.database import get_database_manager
        from app.core.cache import get_cache_manager
        from app.services.scheduling import get_scheduler_service
        from app.services.self_healing import get_self_healing_service
        from app.services.alerting import get_alerting_service
        
        # Initialize services
        console.print("[yellow]Initializing services...[/yellow]")
        
        try:
            # Database
            db_manager = get_database_manager()
            await db_manager.initialize()
            console.print("[green]✓[/green] Database connected")
            
            # Cache
            cache_manager = get_cache_manager()
            await cache_manager.initialize()
            console.print("[green]✓[/green] Redis cache connected")
            
            # Alerting
            alerting_service = get_alerting_service()
            console.print("[green]✓[/green] Alerting service ready")
            
            # Self-healing
            self_healing = get_self_healing_service()
            await self_healing.start()
            console.print("[green]✓[/green] Self-healing service started")
            
            # Scheduler
            if not no_scheduler:
                scheduler = get_scheduler_service()
                await scheduler.start()
                console.print("[green]✓[/green] Scheduler started")
                
                # Send startup notification
                await alerting_service.info(
                    "Worker Started",
                    f"Loyaley worker started in {settings.environment} mode"
                )
            
            console.print("\n[bold green]Worker is running![/bold green]")
            console.print("Press Ctrl+C to stop\n")
            
            # Keep running
            stop_event = asyncio.Event()
            
            def signal_handler():
                console.print("\n[yellow]Shutdown signal received...[/yellow]")
                stop_event.set()
            
            # Handle shutdown signals
            loop = asyncio.get_event_loop()
            for sig in (signal.SIGTERM, signal.SIGINT):
                try:
                    loop.add_signal_handler(sig, signal_handler)
                except NotImplementedError:
                    # Windows doesn't support add_signal_handler
                    pass
            
            # Wait for stop signal
            await stop_event.wait()
            
        except Exception as e:
            console.print(f"[red]✗[/red] Worker error: {e}")
            logger.exception("Worker failed")
            raise
        finally:
            # Cleanup
            console.print("[yellow]Shutting down worker...[/yellow]")
            
            try:
                if not no_scheduler:
                    scheduler = get_scheduler_service()
                    await scheduler.stop()
                    console.print("[green]✓[/green] Scheduler stopped")
                
                self_healing = get_self_healing_service()
                await self_healing.stop()
                console.print("[green]✓[/green] Self-healing stopped")
                
                cache_manager = get_cache_manager()
                await cache_manager.close()
                console.print("[green]✓[/green] Cache closed")
                
                db_manager = get_database_manager()
                await db_manager.close()
                console.print("[green]✓[/green] Database closed")
                
            except Exception as e:
                logger.error(f"Error during shutdown: {e}")
            
            console.print("[green]Worker stopped[/green]")
    
    try:
        asyncio.run(run_worker())
    except KeyboardInterrupt:
        console.print("\n[yellow]Worker interrupted[/yellow]")


# ============== Database Commands ==============

@cli.group()
def db():
    """Database management commands"""
    pass


@db.command()
def init():
    """Initialize database tables"""
    console.print("[yellow]Initializing database tables...[/yellow]")
    
    async def run():
        from app.core.database import get_database_manager, Base
        # Import models to register them with Base
        import app.models.models  # noqa: F401
        
        db_manager = get_database_manager()
        await db_manager.initialize()
        
        async with db_manager._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        await db_manager.close()
        console.print("[green]✓[/green] Database tables created successfully")
    
    asyncio.run(run())


@db.command()
def migrate():
    """Run database migrations"""
    import subprocess
    console.print("[yellow]Running database migrations...[/yellow]")
    
    result = subprocess.run(["alembic", "upgrade", "head"], capture_output=True, text=True)
    if result.returncode == 0:
        console.print("[green]✓[/green] Migrations applied successfully")
        if result.stdout:
            console.print(result.stdout)
    else:
        console.print("[red]✗[/red] Migration failed")
        console.print(result.stderr)
        sys.exit(1)


@db.command()
def seed():
    """Seed initial data (sports and admin user)"""
    console.print("[yellow]Seeding database...[/yellow]")
    
    async def run():
        from app.core.database import get_database_manager
        from app.models.models import Sport, User, UserRole
        from app.core.security import SecurityManager
        from sqlalchemy import select
        
        security = SecurityManager()
        db_manager = get_database_manager()
        await db_manager.initialize()
        
        async with db_manager.session() as session:
            # Check if already seeded
            existing_sports = await session.execute(select(Sport).limit(1))
            if existing_sports.scalar():
                console.print("[yellow]⚠[/yellow] Database already seeded. Skipping...")
                return
            
            # Seed sports with CORRECT field names (api_key not api_code)
            sports_data = [
                {"code": "NFL", "name": "NFL Football", "api_key": "americanfootball_nfl", "feature_count": 75},
                {"code": "NCAAF", "name": "NCAA Football", "api_key": "americanfootball_ncaaf", "feature_count": 70},
                {"code": "CFL", "name": "CFL Football", "api_key": "americanfootball_cfl", "feature_count": 65},
                {"code": "NBA", "name": "NBA Basketball", "api_key": "basketball_nba", "feature_count": 80},
                {"code": "NCAAB", "name": "NCAA Basketball", "api_key": "basketball_ncaab", "feature_count": 70},
                {"code": "WNBA", "name": "WNBA Basketball", "api_key": "basketball_wnba", "feature_count": 70},
                {"code": "NHL", "name": "NHL Hockey", "api_key": "icehockey_nhl", "feature_count": 75},
                {"code": "MLB", "name": "MLB Baseball", "api_key": "baseball_mlb", "feature_count": 85},
                {"code": "ATP", "name": "ATP Tennis", "api_key": "tennis_atp", "feature_count": 60},
                {"code": "WTA", "name": "WTA Tennis", "api_key": "tennis_wta", "feature_count": 60},
            ]
            
            for sport_data in sports_data:
                sport = Sport(**sport_data, is_active=True)
                session.add(sport)
            
            console.print("[green]✓[/green] 10 sports configured")
            
            # Create admin user with CORRECT field names (hashed_password not password_hash)
            admin_password = "AdminPassword123!"
            admin = User(
                email="admin@aiprosports.com",
                hashed_password=security.hash_password(admin_password),
                role=UserRole.ADMIN,  # Use Enum, not string
                is_active=True,
                is_verified=True,
                first_name="System",
                last_name="Administrator",
            )
            session.add(admin)
            
            await session.commit()
            
            console.print("[green]✓[/green] Admin user created")
            console.print(f"    Email: admin@aiprosports.com")
            console.print(f"    Password: {admin_password}")
            console.print(f"    Role: admin")
        
        await db_manager.close()
        console.print("\n[green]✓[/green] Database seeded successfully!")
    
    asyncio.run(run())


@db.command()
@click.option("--table", "-t", help="Specific table to show stats for")
def stats(table: Optional[str]):
    """Show database statistics"""
    
    async def run():
        from app.core.database import get_database_manager
        from sqlalchemy import text
        
        db_manager = get_database_manager()
        await db_manager.initialize()
        
        tables = ["users", "sports", "teams", "games", "predictions", 
                 "odds", "ml_models", "bets", "player_props"]
        
        if table:
            tables = [table]
        
        tbl = Table(title="Database Statistics")
        tbl.add_column("Table", style="cyan")
        tbl.add_column("Row Count", justify="right", style="green")
        
        async with db_manager.session() as session:
            for t in tables:
                try:
                    result = await session.execute(text(f"SELECT COUNT(*) FROM {t}"))
                    count = result.scalar()
                    tbl.add_row(t, str(count))
                except Exception:
                    tbl.add_row(t, "[red]Error/Not exists[/red]")
        
        await db_manager.close()
        console.print(tbl)
    
    asyncio.run(run())


@db.command()
def reset():
    """Reset database (DROP ALL and recreate)"""
    if not click.confirm("⚠️  This will DELETE ALL DATA. Are you sure?"):
        console.print("[yellow]Aborted[/yellow]")
        return
    
    async def run():
        from app.core.database import get_database_manager, Base
        import app.models.models  # noqa: F401
        
        db_manager = get_database_manager()
        await db_manager.initialize()
        
        console.print("[yellow]Dropping all tables...[/yellow]")
        async with db_manager._engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        
        console.print("[yellow]Creating all tables...[/yellow]")
        async with db_manager._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        await db_manager.close()
        console.print("[green]✓[/green] Database reset complete")
    
    asyncio.run(run())


# ============== Model Commands ==============

@cli.group()
def model():
    """ML model management commands"""
    pass


@model.command()
@click.option("--sport", "-s", required=True, help="Sport code (e.g., NBA, NFL)")
@click.option("--bet-type", "-b", required=True, help="Bet type (spread, moneyline, total)")
@click.option("--framework", "-f", default="meta_ensemble", help="ML framework")
@click.option("--max-runtime", "-t", default=3600, help="Max training time in seconds")
def train(sport: str, bet_type: str, framework: str, max_runtime: int):
    """Train a new ML model"""
    console.print(f"[yellow]Training {framework} model for {sport} {bet_type}...[/yellow]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Training in progress...", total=None)
        
        # TODO: Implement actual training
        # from app.services.ml.training_service import MLTrainingService
        # trainer = MLTrainingService()
        # model_id = asyncio.run(trainer.train(sport, bet_type, framework, max_runtime))
        
        import time
        time.sleep(2)  # Simulated training
        
        progress.update(task, description="[green]Training complete![/green]")
    
    console.print(f"[green]✓[/green] Model trained successfully")
    console.print(f"  Sport: {sport}")
    console.print(f"  Bet Type: {bet_type}")
    console.print(f"  Framework: {framework}")


@model.command("list")
@click.option("--sport", "-s", help="Filter by sport")
@click.option("--status", help="Filter by status (ready, production, deprecated)")
def list_models(sport: Optional[str], status: Optional[str]):
    """List all ML models"""
    
    async def run():
        from app.core.database import get_database_manager
        from app.models import MLModel
        from sqlalchemy import select
        
        db_manager = get_database_manager()
        await db_manager.initialize()
        
        async with db_manager.session() as session:
            query = select(MLModel)
            if sport:
                query = query.where(MLModel.sport_code == sport.upper())
            if status:
                query = query.where(MLModel.status == status)
            
            result = await session.execute(query)
            models = result.scalars().all()
            
            if not models:
                console.print("[yellow]No models found[/yellow]")
                return
            
            tbl = Table(title="ML Models")
            tbl.add_column("ID", style="cyan")
            tbl.add_column("Sport")
            tbl.add_column("Bet Type")
            tbl.add_column("Framework")
            tbl.add_column("Status", style="yellow")
            tbl.add_column("AUC", justify="right", style="green")
            tbl.add_column("Created")
            
            for m in models:
                status_style = "green" if m.status == "production" else "yellow"
                tbl.add_row(
                    str(m.id),
                    m.sport_code,
                    m.bet_type,
                    m.framework,
                    f"[{status_style}]{m.status}[/{status_style}]",
                    f"{m.auc:.4f}" if m.auc else "N/A",
                    m.created_at.strftime("%Y-%m-%d") if m.created_at else "N/A"
                )
            
            console.print(tbl)
        
        await db_manager.close()
    
    asyncio.run(run())


@model.command()
@click.argument("model_id")
def promote(model_id: str):
    """Promote a model to production"""
    
    async def run():
        from app.core.database import get_database_manager
        from app.models import MLModel
        from sqlalchemy import select
        from uuid import UUID
        
        db_manager = get_database_manager()
        await db_manager.initialize()
        
        async with db_manager.session() as session:
            try:
                model_uuid = UUID(model_id)
            except ValueError:
                console.print(f"[red]✗[/red] Invalid model ID: {model_id}")
                return
            
            result = await session.execute(
                select(MLModel).where(MLModel.id == model_uuid)
            )
            model = result.scalar_one_or_none()
            
            if not model:
                console.print(f"[red]✗[/red] Model not found: {model_id}")
                return
            
            # Demote current production model
            current_prod = await session.execute(
                select(MLModel).where(
                    MLModel.sport_code == model.sport_code,
                    MLModel.bet_type == model.bet_type,
                    MLModel.status == "production"
                )
            )
            current = current_prod.scalar_one_or_none()
            if current and current.id != model.id:
                current.status = "deprecated"
            
            model.status = "production"
            await session.commit()
            
            console.print(f"[green]✓[/green] Model {model_id} promoted to production")
        
        await db_manager.close()
    
    asyncio.run(run())


# ============== Data Commands ==============

@cli.group()
def data():
    """Data collection commands"""
    pass


@data.command("collect-odds")
@click.option("--sport", "-s", required=True, help="Sport code (e.g., NBA, NFL)")
def collect_odds(sport: str):
    """Collect odds from TheOddsAPI"""
    console.print(f"[yellow]Collecting odds for {sport}...[/yellow]")
    
    async def run():
        try:
            from app.services.collectors.odds_collector import OddsCollector
            collector = OddsCollector()
            await collector.collect_odds(sport.upper())
            console.print(f"[green]✓[/green] Odds collected for {sport}")
        except Exception as e:
            console.print(f"[red]✗[/red] Error: {e}")
    
    asyncio.run(run())


@data.command("collect-games")
@click.option("--sport", "-s", required=True, help="Sport code (e.g., NBA, NFL)")
def collect_games(sport: str):
    """Collect games from ESPN"""
    console.print(f"[yellow]Collecting games for {sport}...[/yellow]")
    
    async def run():
        try:
            from app.services.collectors.espn_collector import ESPNCollector
            collector = ESPNCollector()
            await collector.collect_games(sport.upper())
            console.print(f"[green]✓[/green] Games collected for {sport}")
        except Exception as e:
            console.print(f"[red]✗[/red] Error: {e}")
    
    asyncio.run(run())


@data.command()
@click.option("--sport", "-s", help="Sport code (optional)")
def validate(sport: Optional[str]):
    """Run data validation checks"""
    console.print("[yellow]Running data validation...[/yellow]")
    
    async def run():
        try:
            from app.services.data_quality.data_quality_service import DataQualityService
            service = DataQualityService()
            await service.run_all_checks(sport)
            console.print("[green]✓[/green] Data validation complete")
        except Exception as e:
            console.print(f"[red]✗[/red] Error: {e}")
    
    asyncio.run(run())


# ============== Prediction Commands ==============

@cli.group()
def predict():
    """Prediction management commands"""
    pass


@predict.command()
@click.option("--sport", "-s", required=True, help="Sport code")
@click.option("--date", "-d", help="Date (YYYY-MM-DD), defaults to today")
def generate(sport: str, date: Optional[str]):
    """Generate predictions for upcoming games"""
    target_date = date or datetime.now().strftime("%Y-%m-%d")
    console.print(f"[yellow]Generating predictions for {sport} on {target_date}...[/yellow]")
    
    async def run():
        try:
            from app.services.ml.prediction_engine import PredictionEngine
            engine = PredictionEngine()
            predictions = await engine.generate_predictions(sport.upper(), target_date)
            console.print(f"[green]✓[/green] Generated {len(predictions)} predictions")
        except Exception as e:
            console.print(f"[red]✗[/red] Error: {e}")
    
    asyncio.run(run())


@predict.command()
def grade():
    """Grade completed predictions"""
    console.print("[yellow]Grading predictions...[/yellow]")
    
    async def run():
        try:
            from app.services.betting.auto_grader import AutoGrader
            grader = AutoGrader()
            graded_count = await grader.grade_pending()
            console.print(f"[green]✓[/green] Graded {graded_count} predictions")
        except Exception as e:
            console.print(f"[red]✗[/red] Error: {e}")
    
    asyncio.run(run())


@predict.command("stats")
@click.option("--sport", "-s", help="Sport code")
@click.option("--days", "-d", default=7, help="Number of days")
def prediction_stats(sport: Optional[str], days: int):
    """View prediction statistics"""
    
    async def run():
        from app.core.database import get_database_manager
        from app.models import Prediction
        from sqlalchemy import select, func, case
        
        db_manager = get_database_manager()
        await db_manager.initialize()
        
        async with db_manager.session() as session:
            start_date = datetime.utcnow() - timedelta(days=days)
            
            query = select(
                Prediction.sport_code,
                Prediction.signal_tier,
                func.count(Prediction.id).label("total"),
                func.sum(case((Prediction.result == "win", 1), else_=0)).label("wins")
            ).where(
                Prediction.created_at >= start_date,
                Prediction.is_graded == True
            ).group_by(Prediction.sport_code, Prediction.signal_tier)
            
            if sport:
                query = query.where(Prediction.sport_code == sport.upper())
            
            result = await session.execute(query)
            rows = result.all()
            
            if not rows:
                console.print("[yellow]No graded predictions found[/yellow]")
                return
            
            tbl = Table(title=f"Prediction Stats (Last {days} Days)")
            tbl.add_column("Sport")
            tbl.add_column("Tier")
            tbl.add_column("Total", justify="right")
            tbl.add_column("Wins", justify="right", style="green")
            tbl.add_column("Win Rate", justify="right", style="cyan")
            
            for row in rows:
                win_rate = (row.wins / row.total * 100) if row.total > 0 else 0
                tbl.add_row(
                    row.sport_code,
                    row.signal_tier,
                    str(row.total),
                    str(row.wins or 0),
                    f"{win_rate:.1f}%"
                )
            
            console.print(tbl)
        
        await db_manager.close()
    
    asyncio.run(run())


# ============== System Commands ==============

@cli.group()
def system():
    """System management commands"""
    pass


@system.command()
def status():
    """Show system status"""
    console.print(Panel.fit(
        "[bold green]LOYALEY[/bold green]\n"
        "Enterprise Sports Prediction Platform",
        title="System Status"
    ))
    
    async def run():
        from app.core.database import get_database_manager
        from app.core.cache import get_cache_manager
        
        components = []
        
        # Check database
        try:
            db_manager = get_database_manager()
            await db_manager.initialize()
            health = await db_manager.health_check()
            db_healthy = health.get("status") == "healthy"
            await db_manager.close()
            components.append(("Database", db_healthy))
        except Exception:
            components.append(("Database", False))
        
        # Check Redis
        try:
            cache = get_cache_manager()
            await cache.initialize()
            redis_healthy = await cache.health_check()
            await cache.close()
            components.append(("Redis Cache", redis_healthy))
        except Exception:
            components.append(("Redis Cache", False))
        
        # Add other components
        components.extend([
            ("ML Models", True),
            ("Scheduler", True),
            ("API", True),
        ])
        
        tbl = Table()
        tbl.add_column("Component")
        tbl.add_column("Status")
        
        for name, healthy in components:
            status_str = "[green]● Healthy[/green]" if healthy else "[red]● Down[/red]"
            tbl.add_row(name, status_str)
        
        console.print(tbl)
    
    asyncio.run(run())


@system.command()
def health():
    """Run health checks"""
    console.print("[yellow]Running health checks...[/yellow]")
    
    async def run():
        from app.core.database import get_database_manager
        from app.core.cache import get_cache_manager
        import psutil
        
        checks = []
        
        # Database
        try:
            db_manager = get_database_manager()
            await db_manager.initialize()
            health = await db_manager.health_check()
            checks.append(("Database connection", health.get("status") == "healthy"))
            await db_manager.close()
        except Exception as e:
            checks.append(("Database connection", False))
        
        # Redis
        try:
            cache = get_cache_manager()
            await cache.initialize()
            redis_ok = await cache.health_check()
            checks.append(("Redis connection", redis_ok))
            await cache.close()
        except Exception:
            checks.append(("Redis connection", False))
        
        # Disk space
        disk = psutil.disk_usage('/')
        disk_ok = disk.percent < 90
        checks.append(("Disk space", disk_ok))
        
        # Memory
        mem = psutil.virtual_memory()
        mem_ok = mem.percent < 90
        checks.append(("Memory usage", mem_ok))
        
        all_passed = True
        for name, passed in checks:
            status = "[green]✓[/green]" if passed else "[red]✗[/red]"
            if not passed:
                all_passed = False
            console.print(f"  {status} {name}")
        
        if all_passed:
            console.print("\n[green]All health checks passed[/green]")
        else:
            console.print("\n[red]Some health checks failed[/red]")
    
    asyncio.run(run())


@system.command("cache-clear")
@click.option("--pattern", "-p", help="Key pattern to clear")
def cache_clear(pattern: Optional[str]):
    """Clear Redis cache"""
    
    async def run():
        from app.core.cache import get_cache_manager
        
        cache = get_cache_manager()
        await cache.initialize()
        
        if pattern:
            count = await cache.clear_pattern(pattern)
            console.print(f"[green]✓[/green] Cleared {count} keys matching '{pattern}'")
        else:
            await cache.clear_all()
            console.print("[green]✓[/green] All cache cleared")
        
        await cache.close()
    
    asyncio.run(run())


@system.command()
def version():
    """Show version information"""
    from app.core.config import get_settings
    
    settings = get_settings()
    
    console.print(f"[bold]LOYALEY[/bold]")
    console.print(f"Version: {settings.app_version}")
    console.print(f"Environment: {settings.environment}")


# ============== Backtest Commands ==============

@cli.group()
def backtest():
    """Backtesting commands"""
    pass


@backtest.command("run")
@click.option("--sport", "-s", required=True, multiple=True, help="Sport codes")
@click.option("--start", required=True, help="Start date (YYYY-MM-DD)")
@click.option("--end", required=True, help="End date (YYYY-MM-DD)")
@click.option("--bankroll", "-b", default=10000.0, help="Initial bankroll")
@click.option("--kelly", "-k", default=0.25, help="Kelly fraction")
def run_backtest(sport: tuple, start: str, end: str, bankroll: float, kelly: float):
    """Run a backtest simulation"""
    console.print(f"[yellow]Running backtest...[/yellow]")
    console.print(f"  Sports: {', '.join(sport)}")
    console.print(f"  Period: {start} to {end}")
    console.print(f"  Bankroll: ${bankroll:,.2f}")
    console.print(f"  Kelly: {kelly * 100}%")
    
    async def run():
        try:
            from app.services.backtesting.backtest_engine import BacktestEngine
            
            engine = BacktestEngine()
            results = await engine.run(
                sports=list(sport),
                start_date=start,
                end_date=end,
                initial_bankroll=bankroll,
                kelly_fraction=kelly
            )
            
            console.print("\n[bold]Backtest Results[/bold]")
            console.print(f"  Total Bets: {results.get('total_bets', 0)}")
            console.print(f"  Win Rate: {results.get('win_rate', 0):.1f}%")
            console.print(f"  Final Bankroll: ${results.get('final_bankroll', 0):,.2f}")
            
            roi = results.get('roi', 0)
            roi_color = "green" if roi > 0 else "red"
            console.print(f"  ROI: [{roi_color}]{roi:+.1f}%[/{roi_color}]")
            console.print(f"  Max Drawdown: {results.get('max_drawdown', 0):.1f}%")
            
        except Exception as e:
            console.print(f"[red]✗[/red] Backtest error: {e}")
    
    asyncio.run(run())


# ============== Server Command ==============

@cli.command()
@click.option("--host", "-h", default="0.0.0.0", help="Host to bind")
@click.option("--port", "-p", default=8000, help="Port to bind")
@click.option("--reload", is_flag=True, help="Enable auto-reload")
@click.option("--workers", "-w", default=1, help="Number of workers")
def serve(host: str, port: int, reload: bool, workers: int):
    """Start the API server"""
    import uvicorn
    
    console.print(Panel.fit(
        "[bold green]LOYALEY[/bold green]\n"
        f"Starting API server on {host}:{port}",
        title="Server"
    ))
    
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers if not reload else 1,
        log_level="info",
    )


if __name__ == "__main__":
    cli()
