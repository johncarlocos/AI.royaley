#!/usr/bin/env python3
"""
ROYALEY - Backup Manager
Phase 4: Enterprise Operations - Backup & Recovery

Automated backup system for:
- PostgreSQL database dumps
- ML model files
- Configuration files

Storage: /sda-disk/backups/

Usage:
    # Full backup
    python backup_manager.py --full
    
    # Database only
    python backup_manager.py --database
    
    # Models only
    python backup_manager.py --models
    
    # Restore from backup
    python backup_manager.py --restore --file /path/to/backup.sql.gz
"""

import asyncio
import argparse
import gzip
import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.panel import Panel

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import settings

console = Console()
logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


@dataclass
class BackupInfo:
    """Backup information."""
    name: str
    path: Path
    size_mb: float
    created_at: datetime
    backup_type: str  # database, models, config, full


class BackupManager:
    """
    Enterprise backup manager.
    
    Backup Schedule:
    - Daily: PostgreSQL dump, compressed
    - Weekly: Full system backup (DB + models + config)
    - Monthly: Archived monthly backup
    
    Retention:
    - Daily: 7 days
    - Weekly: 4 weeks
    - Monthly: 12 months
    """
    
    def __init__(self):
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create backup directories."""
        Path(settings.DB_BACKUPS_PATH).mkdir(parents=True, exist_ok=True)
        Path(settings.MODEL_BACKUPS_PATH).mkdir(parents=True, exist_ok=True)
        Path(settings.CONFIG_BACKUPS_PATH).mkdir(parents=True, exist_ok=True)
        Path(settings.DAILY_ARCHIVES_PATH).mkdir(parents=True, exist_ok=True)
        Path(settings.WEEKLY_ARCHIVES_PATH).mkdir(parents=True, exist_ok=True)
        Path(settings.MONTHLY_ARCHIVES_PATH).mkdir(parents=True, exist_ok=True)
    
    async def full_backup(self) -> Dict[str, BackupInfo]:
        """Perform full system backup."""
        console.print(Panel(
            "[bold green]Full System Backup[/bold green]\n"
            "Database + Models + Configuration",
            title="Backup Manager"
        ))
        
        results = {}
        
        # Database backup
        db_backup = await self.backup_database()
        if db_backup:
            results["database"] = db_backup
        
        # Models backup
        model_backup = await self.backup_models()
        if model_backup:
            results["models"] = model_backup
        
        # Config backup
        config_backup = await self.backup_config()
        if config_backup:
            results["config"] = config_backup
        
        # Cleanup old backups
        await self.cleanup_old_backups()
        
        self._print_summary(results)
        return results
    
    async def backup_database(self) -> Optional[BackupInfo]:
        """Backup PostgreSQL database."""
        console.print("[yellow]Backing up database...[/yellow]")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"royaley_db_{timestamp}.sql.gz"
        backup_path = Path(settings.DB_BACKUPS_PATH) / backup_name
        
        try:
            # Use pg_dump through Docker
            pg_dump_cmd = [
                "docker", "exec", "royaley_postgres",
                "pg_dump",
                "-U", settings.POSTGRES_USER,
                "-d", settings.POSTGRES_DB,
                "--no-owner",
                "--no-acl",
            ]
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Dumping database...", total=None)
                
                # Run pg_dump and capture output
                result = subprocess.run(
                    pg_dump_cmd,
                    capture_output=True,
                    text=True
                )
                
                if result.returncode != 0:
                    console.print(f"[red]pg_dump failed: {result.stderr}[/red]")
                    return None
                
                progress.update(task, description="Compressing backup...")
                
                # Compress and save
                with gzip.open(backup_path, 'wt', encoding='utf-8') as f:
                    f.write(result.stdout)
            
            size_mb = backup_path.stat().st_size / (1024 * 1024)
            
            backup_info = BackupInfo(
                name=backup_name,
                path=backup_path,
                size_mb=size_mb,
                created_at=datetime.now(),
                backup_type="database"
            )
            
            console.print(f"[green]✓ Database backup: {backup_name} ({size_mb:.2f} MB)[/green]")
            return backup_info
            
        except FileNotFoundError:
            console.print("[red]Docker not found. Running manual pg_dump...[/red]")
            return await self._backup_database_direct()
        except Exception as e:
            console.print(f"[red]Database backup failed: {e}[/red]")
            return None
    
    async def _backup_database_direct(self) -> Optional[BackupInfo]:
        """Direct database backup without Docker."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"royaley_db_{timestamp}.sql.gz"
        backup_path = Path(settings.DB_BACKUPS_PATH) / backup_name
        
        try:
            # Direct pg_dump
            env = os.environ.copy()
            env["PGPASSWORD"] = settings.POSTGRES_PASSWORD
            
            pg_dump_cmd = [
                "pg_dump",
                "-h", settings.POSTGRES_HOST,
                "-p", str(settings.POSTGRES_PORT),
                "-U", settings.POSTGRES_USER,
                "-d", settings.POSTGRES_DB,
                "--no-owner",
                "--no-acl",
            ]
            
            result = subprocess.run(
                pg_dump_cmd,
                capture_output=True,
                text=True,
                env=env
            )
            
            if result.returncode != 0:
                console.print(f"[red]pg_dump failed: {result.stderr}[/red]")
                return None
            
            with gzip.open(backup_path, 'wt', encoding='utf-8') as f:
                f.write(result.stdout)
            
            size_mb = backup_path.stat().st_size / (1024 * 1024)
            
            return BackupInfo(
                name=backup_name,
                path=backup_path,
                size_mb=size_mb,
                created_at=datetime.now(),
                backup_type="database"
            )
            
        except Exception as e:
            console.print(f"[red]Direct backup failed: {e}[/red]")
            return None
    
    async def backup_models(self) -> Optional[BackupInfo]:
        """Backup ML models."""
        console.print("[yellow]Backing up models...[/yellow]")
        
        models_path = Path(settings.MODELS_PATH)
        
        if not models_path.exists() or not any(models_path.iterdir()):
            console.print("[yellow]No models to backup[/yellow]")
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"royaley_models_{timestamp}.tar.gz"
        backup_path = Path(settings.MODEL_BACKUPS_PATH) / backup_name
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Compressing models...", total=None)
                
                # Create tar.gz archive
                shutil.make_archive(
                    str(backup_path).replace('.tar.gz', ''),
                    'gztar',
                    root_dir=models_path.parent,
                    base_dir=models_path.name
                )
            
            size_mb = backup_path.stat().st_size / (1024 * 1024)
            
            backup_info = BackupInfo(
                name=backup_name,
                path=backup_path,
                size_mb=size_mb,
                created_at=datetime.now(),
                backup_type="models"
            )
            
            console.print(f"[green]✓ Models backup: {backup_name} ({size_mb:.2f} MB)[/green]")
            return backup_info
            
        except Exception as e:
            console.print(f"[red]Models backup failed: {e}[/red]")
            return None
    
    async def backup_config(self) -> Optional[BackupInfo]:
        """Backup configuration files."""
        console.print("[yellow]Backing up configuration...[/yellow]")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"royaley_config_{timestamp}.tar.gz"
        backup_path = Path(settings.CONFIG_BACKUPS_PATH) / backup_name
        
        # Files to backup
        config_files = [
            ".env",
            ".env.production",
            "docker-compose.yml",
            "config/prometheus.yml",
            "alembic.ini",
        ]
        
        try:
            import tarfile
            
            base_path = Path(settings.NVME0_BASE_PATH)
            
            with tarfile.open(backup_path, "w:gz") as tar:
                for config_file in config_files:
                    file_path = base_path / config_file
                    if file_path.exists():
                        tar.add(file_path, arcname=config_file)
            
            size_mb = backup_path.stat().st_size / (1024 * 1024)
            
            backup_info = BackupInfo(
                name=backup_name,
                path=backup_path,
                size_mb=size_mb,
                created_at=datetime.now(),
                backup_type="config"
            )
            
            console.print(f"[green]✓ Config backup: {backup_name} ({size_mb:.2f} MB)[/green]")
            return backup_info
            
        except Exception as e:
            console.print(f"[red]Config backup failed: {e}[/red]")
            return None
    
    async def restore_database(self, backup_file: str) -> bool:
        """Restore database from backup."""
        backup_path = Path(backup_file)
        
        if not backup_path.exists():
            console.print(f"[red]Backup file not found: {backup_file}[/red]")
            return False
        
        console.print(Panel(
            f"[bold yellow]Database Restore[/bold yellow]\n"
            f"From: {backup_file}\n"
            f"[red]WARNING: This will overwrite existing data![/red]",
            title="Restore"
        ))
        
        # Confirm
        confirm = console.input("[yellow]Type 'RESTORE' to confirm: [/yellow]")
        if confirm != "RESTORE":
            console.print("[yellow]Restore cancelled[/yellow]")
            return False
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Restoring database...", total=None)
                
                # Decompress if gzipped
                if backup_path.suffix == '.gz':
                    with gzip.open(backup_path, 'rt', encoding='utf-8') as f:
                        sql_content = f.read()
                else:
                    with open(backup_path, 'r') as f:
                        sql_content = f.read()
                
                # Restore through Docker
                restore_cmd = [
                    "docker", "exec", "-i", "royaley_postgres",
                    "psql",
                    "-U", settings.POSTGRES_USER,
                    "-d", settings.POSTGRES_DB,
                ]
                
                result = subprocess.run(
                    restore_cmd,
                    input=sql_content,
                    capture_output=True,
                    text=True
                )
                
                if result.returncode != 0:
                    console.print(f"[red]Restore failed: {result.stderr}[/red]")
                    return False
            
            console.print("[green]✓ Database restored successfully[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]Restore failed: {e}[/red]")
            return False
    
    async def cleanup_old_backups(self):
        """Remove old backups based on retention policy."""
        console.print("[yellow]Cleaning up old backups...[/yellow]")
        
        now = datetime.now()
        
        # Daily backups: keep 7 days
        daily_cutoff = now - timedelta(days=settings.BACKUP_RETENTION_DAILY)
        self._cleanup_directory(
            Path(settings.DB_BACKUPS_PATH),
            daily_cutoff,
            "daily database"
        )
        
        # Model backups: keep 4 weeks
        model_cutoff = now - timedelta(weeks=settings.BACKUP_RETENTION_WEEKLY)
        self._cleanup_directory(
            Path(settings.MODEL_BACKUPS_PATH),
            model_cutoff,
            "model"
        )
        
        # Config backups: keep 4 weeks
        self._cleanup_directory(
            Path(settings.CONFIG_BACKUPS_PATH),
            model_cutoff,
            "config"
        )
    
    def _cleanup_directory(
        self,
        directory: Path,
        cutoff: datetime,
        backup_type: str,
    ):
        """Remove files older than cutoff date."""
        if not directory.exists():
            return
        
        removed = 0
        for file in directory.iterdir():
            if file.is_file():
                mtime = datetime.fromtimestamp(file.stat().st_mtime)
                if mtime < cutoff:
                    file.unlink()
                    removed += 1
        
        if removed:
            console.print(f"  [dim]Removed {removed} old {backup_type} backups[/dim]")
    
    def list_backups(self) -> Dict[str, List[BackupInfo]]:
        """List all available backups."""
        backups = {
            "database": [],
            "models": [],
            "config": [],
        }
        
        # Database backups
        db_path = Path(settings.DB_BACKUPS_PATH)
        if db_path.exists():
            for file in sorted(db_path.iterdir(), reverse=True):
                if file.is_file():
                    backups["database"].append(BackupInfo(
                        name=file.name,
                        path=file,
                        size_mb=file.stat().st_size / (1024 * 1024),
                        created_at=datetime.fromtimestamp(file.stat().st_mtime),
                        backup_type="database"
                    ))
        
        # Model backups
        model_path = Path(settings.MODEL_BACKUPS_PATH)
        if model_path.exists():
            for file in sorted(model_path.iterdir(), reverse=True):
                if file.is_file():
                    backups["models"].append(BackupInfo(
                        name=file.name,
                        path=file,
                        size_mb=file.stat().st_size / (1024 * 1024),
                        created_at=datetime.fromtimestamp(file.stat().st_mtime),
                        backup_type="models"
                    ))
        
        # Config backups
        config_path = Path(settings.CONFIG_BACKUPS_PATH)
        if config_path.exists():
            for file in sorted(config_path.iterdir(), reverse=True):
                if file.is_file():
                    backups["config"].append(BackupInfo(
                        name=file.name,
                        path=file,
                        size_mb=file.stat().st_size / (1024 * 1024),
                        created_at=datetime.fromtimestamp(file.stat().st_mtime),
                        backup_type="config"
                    ))
        
        return backups
    
    def _print_summary(self, results: Dict[str, BackupInfo]):
        """Print backup summary."""
        table = Table(title="Backup Summary")
        table.add_column("Type", style="cyan")
        table.add_column("File", style="green")
        table.add_column("Size (MB)", style="yellow")
        table.add_column("Location", style="blue")
        
        total_size = 0.0
        
        for backup_type, info in results.items():
            table.add_row(
                backup_type,
                info.name,
                f"{info.size_mb:.2f}",
                str(info.path.parent)
            )
            total_size += info.size_mb
        
        table.add_row(
            "[bold]TOTAL[/bold]",
            "-",
            f"[bold]{total_size:.2f}[/bold]",
            "-"
        )
        
        console.print(table)


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Royaley Backup Manager"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Full system backup"
    )
    parser.add_argument(
        "--database",
        action="store_true",
        help="Database backup only"
    )
    parser.add_argument(
        "--models",
        action="store_true",
        help="Models backup only"
    )
    parser.add_argument(
        "--config",
        action="store_true",
        help="Config backup only"
    )
    parser.add_argument(
        "--restore",
        action="store_true",
        help="Restore from backup"
    )
    parser.add_argument(
        "--file", "-f",
        type=str,
        help="Backup file path (for restore)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available backups"
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Clean up old backups"
    )
    
    args = parser.parse_args()
    
    manager = BackupManager()
    
    if args.list:
        backups = manager.list_backups()
        
        for backup_type, items in backups.items():
            if items:
                console.print(f"\n[bold]{backup_type.upper()} Backups:[/bold]")
                for item in items[:5]:  # Show last 5
                    console.print(f"  • {item.name} ({item.size_mb:.2f} MB) - {item.created_at}")
        return
    
    if args.restore:
        if not args.file:
            console.print("[red]Specify backup file with --file[/red]")
            return
        await manager.restore_database(args.file)
        return
    
    if args.cleanup:
        await manager.cleanup_old_backups()
        return
    
    if args.full:
        await manager.full_backup()
    elif args.database:
        await manager.backup_database()
    elif args.models:
        await manager.backup_models()
    elif args.config:
        await manager.backup_config()
    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
