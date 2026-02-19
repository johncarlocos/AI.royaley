#!/usr/bin/env python3
"""
ROYALEY - Database → HDD Bulk Export
=====================================

Exports ALL database tables to compressed CSV files on the 16TB HDD.
This ensures every piece of collected data (from all 27 collectors)
is archived on HDD regardless of which HTTP path each collector uses.

Storage Target: 3TB+ on /app/raw-data (mounted from /sda-disk/raw-data)

Usage:
    # Full export (all tables, all data):
    python scripts/db_to_hdd_export.py --full

    # Incremental export (new data since last export):
    python scripts/db_to_hdd_export.py --incremental

    # Export specific table:
    python scripts/db_to_hdd_export.py --table odds

    # Show current status:
    python scripts/db_to_hdd_export.py --status

    # Schedule (add to data_collector.py or cron):
    # Runs daily at 2 AM - incremental export of new data
"""

import asyncio
import argparse
import csv
import gzip
import io
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("royaley.db_export")

HDD_EXPORT_PATH = Path("/app/raw-data/db-exports")


# =============================================================================
# TABLE DEFINITIONS: what to export, how to partition
# =============================================================================

EXPORT_TABLES = [
    # ── HIGH VOLUME (these will be 90%+ of the 3TB) ──────────────────
    {
        "table": "odds",
        "query": """
            SELECT o.*, s.code as sport_code, g.scheduled_at, g.external_id as game_external_id,
                   ht.name as home_team, at.name as away_team,
                   sb.name as sportsbook_name
            FROM odds o
            JOIN games g ON o.game_id = g.id
            JOIN sports s ON g.sport_id = s.id
            LEFT JOIN teams ht ON g.home_team_id = ht.id
            LEFT JOIN teams at ON g.away_team_id = at.id
            LEFT JOIN sportsbooks sb ON o.sportsbook_id = sb.id
            {where}
            ORDER BY g.scheduled_at, o.recorded_at
        """,
        "partition_by": "sport_code",
        "date_col": "o.recorded_at",
        "description": "Odds from 40+ sportsbooks",
        "estimated_size": "500GB-1TB",
    },
    {
        "table": "odds_movements",
        "query": """
            SELECT om.*, s.code as sport_code, g.scheduled_at, g.external_id as game_external_id
            FROM odds_movements om
            JOIN games g ON om.game_id = g.id
            JOIN sports s ON g.sport_id = s.id
            {where}
            ORDER BY om.detected_at
        """,
        "partition_by": "sport_code",
        "date_col": "om.detected_at",
        "description": "Line movement history",
        "estimated_size": "200-500GB",
    },
    {
        "table": "closing_lines",
        "query": """
            SELECT cl.*, s.code as sport_code, g.scheduled_at, g.external_id as game_external_id
            FROM closing_lines cl
            JOIN games g ON cl.game_id = g.id
            JOIN sports s ON g.sport_id = s.id
            {where}
            ORDER BY cl.recorded_at
        """,
        "partition_by": "sport_code",
        "date_col": "cl.recorded_at",
        "description": "Pinnacle closing lines",
        "estimated_size": "50-100GB",
    },
    # ── MEDIUM VOLUME ─────────────────────────────────────────────────
    {
        "table": "games",
        "query": """
            SELECT g.id, g.sport_id, g.season_id, g.external_id, g.home_team_id, g.away_team_id,
                   g.venue_id, g.scheduled_at, g.status, g.home_score, g.away_score,
                   g.home_rotation, g.away_rotation, g.period, g.clock,
                   g.master_game_id, g.created_at, g.updated_at,
                   s.code as sport_code, s.name as sport_name,
                   ht.name as home_team_name, ht.abbreviation as home_abbr,
                   at.name as away_team_name, at.abbreviation as away_abbr,
                   v.name as venue_name, v.city as venue_city
            FROM games g
            JOIN sports s ON g.sport_id = s.id
            LEFT JOIN teams ht ON g.home_team_id = ht.id
            LEFT JOIN teams at ON g.away_team_id = at.id
            LEFT JOIN venues v ON g.venue_id = v.id
            {where}
            ORDER BY g.scheduled_at
        """,
        "partition_by": "sport_code",
        "date_col": "g.created_at",
        "description": "All games (10 sports, 10 years)",
        "estimated_size": "10-30GB",
    },
    {
        "table": "player_stats",
        "query": """
            SELECT ps.*, s.code as sport_code, g.scheduled_at, g.external_id as game_external_id,
                   p.name as player_name, p.position as player_position
            FROM player_stats ps
            LEFT JOIN games g ON ps.game_id = g.id
            LEFT JOIN sports s ON g.sport_id = s.id
            LEFT JOIN players p ON ps.player_id = p.id
            {where}
            ORDER BY ps.created_at
        """,
        "partition_by": "sport_code",
        "date_col": "ps.created_at",
        "description": "Player-level stats",
        "estimated_size": "50-200GB",
    },
    {
        "table": "team_stats",
        "query": """
            SELECT ts.*, t.name as team_name, t.abbreviation as team_abbr,
                   s.code as sport_code
            FROM team_stats ts
            JOIN teams t ON ts.team_id = t.id
            JOIN sports s ON t.sport_id = s.id
            {where}
            ORDER BY ts.computed_at
        """,
        "partition_by": "sport_code",
        "date_col": "ts.computed_at",
        "description": "Team-level stats",
        "estimated_size": "10-50GB",
    },
    {
        "table": "player_props",
        "query": """
            SELECT pp.*, s.code as sport_code, g.scheduled_at,
                   p.name as player_name
            FROM player_props pp
            JOIN games g ON pp.game_id = g.id
            JOIN sports s ON g.sport_id = s.id
            LEFT JOIN players p ON pp.player_id = p.id
            {where}
            ORDER BY pp.created_at
        """,
        "partition_by": "sport_code",
        "date_col": "pp.created_at",
        "description": "Player prop predictions",
        "estimated_size": "20-50GB",
    },
    # ── LOWER VOLUME ──────────────────────────────────────────────────
    {
        "table": "weather_data",
        "query": """
            SELECT wd.*, g.scheduled_at, g.external_id as game_external_id,
                   s.code as sport_code,
                   v.name as venue_name, v.city as venue_city, v.latitude, v.longitude
            FROM weather_data wd
            LEFT JOIN games g ON wd.game_id = g.id
            LEFT JOIN sports s ON g.sport_id = s.id
            LEFT JOIN venues v ON wd.venue_id = v.id
            {where}
            ORDER BY wd.recorded_at
        """,
        "partition_by": None,
        "date_col": "wd.recorded_at",
        "description": "Weather data",
        "estimated_size": "1-5GB",
    },
    {
        "table": "predictions",
        "query": """
            SELECT p.*, s.code as sport_code, g.scheduled_at, g.external_id as game_external_id,
                   pr.actual_result, pr.profit_loss, pr.clv as closing_line_value,
                   pr.graded_at,
                   ht.name as home_team, at.name as away_team
            FROM predictions p
            JOIN games g ON p.game_id = g.id
            JOIN sports s ON g.sport_id = s.id
            LEFT JOIN prediction_results pr ON p.id = pr.prediction_id
            LEFT JOIN teams ht ON g.home_team_id = ht.id
            LEFT JOIN teams at ON g.away_team_id = at.id
            {where}
            ORDER BY p.created_at
        """,
        "partition_by": "sport_code",
        "date_col": "p.created_at",
        "description": "All predictions with results",
        "estimated_size": "5-20GB",
    },
    {
        "table": "clv_records",
        "query": """
            SELECT clv.*
            FROM clv_records clv
            {where}
            ORDER BY clv.recorded_at
        """,
        "partition_by": "sport_code",
        "date_col": "clv.recorded_at",
        "description": "CLV tracking records",
        "estimated_size": "5-10GB",
    },
    # ── REFERENCE DATA (small but important) ──────────────────────────
    {
        "table": "teams",
        "query": "SELECT t.*, s.code as sport_code FROM teams t JOIN sports s ON t.sport_id = s.id {where} ORDER BY t.name",
        "partition_by": None,
        "date_col": "t.created_at",
        "description": "All teams",
        "estimated_size": "<100MB",
    },
    {
        "table": "players",
        "query": """
            SELECT p.*, t.name as team_name, s.code as sport_code
            FROM players p
            LEFT JOIN teams t ON p.team_id = t.id
            LEFT JOIN sports s ON t.sport_id = s.id
            {where}
            ORDER BY p.name
        """,
        "partition_by": None,
        "date_col": "p.created_at",
        "description": "All players",
        "estimated_size": "<500MB",
    },
    {
        "table": "venues",
        "query": "SELECT * FROM venues {where} ORDER BY name",
        "partition_by": None,
        "date_col": "created_at",
        "description": "Venues",
        "estimated_size": "<50MB",
    },
    {
        "table": "sportsbooks",
        "query": "SELECT * FROM sportsbooks {where} ORDER BY name",
        "partition_by": None,
        "date_col": "created_at",
        "description": "Sportsbooks",
        "estimated_size": "<10MB",
    },
    {
        "table": "elo_history",
        "query": """
            SELECT eh.*, t.name as team_name
            FROM elo_history eh
            JOIN teams t ON eh.team_id = t.id
            {where}
            ORDER BY eh.recorded_at
        """,
        "partition_by": "sport_code",
        "date_col": "eh.recorded_at",
        "description": "Elo rating history",
        "estimated_size": "1-5GB",
    },
    {
        "table": "game_features",
        "query": """
            SELECT gf.id, gf.game_id, gf.feature_version, gf.computed_at,
                   s.code as sport_code, g.scheduled_at
            FROM game_features gf
            JOIN games g ON gf.game_id = g.id
            JOIN sports s ON g.sport_id = s.id
            {where}
            ORDER BY gf.computed_at
        """,
        "partition_by": "sport_code",
        "date_col": "gf.computed_at",
        "description": "Pre-computed ML features",
        "estimated_size": "5-20GB",
    },
]


# =============================================================================
# EXPORT ENGINE
# =============================================================================

class DBExporter:
    """Exports database tables to compressed CSV files on HDD."""
    
    BATCH_SIZE = 50_000  # Rows per query batch
    ROWS_PER_FILE = 500_000  # Max rows per output file
    
    def __init__(self):
        self.export_path = HDD_EXPORT_PATH
        self.export_path.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.export_path / "_export_metadata.json"
        self.metadata = self._load_metadata()
        self.stats = {
            "tables_exported": 0,
            "total_rows": 0,
            "total_bytes": 0,
            "total_files": 0,
            "errors": [],
        }
    
    def _load_metadata(self) -> Dict:
        """Load export metadata (tracks last export timestamps per table)."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file) as f:
                    return json.load(f)
            except Exception:
                pass
        return {"exports": {}, "created": datetime.utcnow().isoformat()}
    
    def _save_metadata(self):
        """Save export metadata."""
        try:
            with open(self.metadata_file, "w") as f:
                json.dump(self.metadata, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    async def export_table(
        self,
        table_def: Dict,
        since: Optional[datetime] = None,
        full: bool = False,
    ) -> Tuple[int, int]:
        """
        Export a single table to compressed CSV files on HDD.
        
        Returns: (rows_exported, bytes_written)
        """
        table_name = table_def["table"]
        date_col = table_def.get("date_col")
        
        logger.info(f"📊 Exporting: {table_name} ({table_def['description']})")
        
        # Build WHERE clause
        where_parts = []
        params = {}
        
        if not full and since and date_col:
            where_parts.append(f"{date_col} >= :since")
            params["since"] = since
        
        where_clause = ""
        if where_parts:
            where_clause = "WHERE " + " AND ".join(where_parts)
        
        query = table_def["query"].replace("{where}", where_clause)
        
        # Execute and write
        try:
            from app.core.database import db_manager
            
            total_rows = 0
            total_bytes = 0
            file_num = 0
            
            async with db_manager.session() as session:
                from sqlalchemy import text
                
                # Count rows first
                count_query = f"SELECT COUNT(*) FROM ({query}) sq"
                result = await session.execute(text(count_query), params)
                row_count = result.scalar()
                
                if row_count == 0:
                    logger.info(f"  ⏭️  {table_name}: 0 rows (skipping)")
                    return 0, 0
                
                logger.info(f"  📦 {table_name}: {row_count:,} rows to export")
                
                # Stream results in batches
                offset = 0
                current_rows = []
                headers = None
                
                while offset < row_count:
                    batch_query = f"{query} LIMIT {self.BATCH_SIZE} OFFSET {offset}"
                    result = await session.execute(text(batch_query), params)
                    rows = result.fetchall()
                    
                    if not rows:
                        break
                    
                    if headers is None:
                        headers = list(result.keys())
                    
                    for row in rows:
                        current_rows.append(dict(zip(headers, row)))
                        
                        # Write file when we hit the limit
                        if len(current_rows) >= self.ROWS_PER_FILE:
                            file_num += 1
                            bytes_written = await self._write_csv_file(
                                table_name, current_rows, headers, file_num, since
                            )
                            total_bytes += bytes_written
                            total_rows += len(current_rows)
                            logger.info(
                                f"  💾 {table_name} part {file_num}: "
                                f"{len(current_rows):,} rows, "
                                f"{bytes_written / (1024**2):.1f} MB"
                            )
                            current_rows = []
                    
                    offset += self.BATCH_SIZE
                
                # Write remaining rows
                if current_rows:
                    file_num += 1
                    bytes_written = await self._write_csv_file(
                        table_name, current_rows, headers, file_num, since
                    )
                    total_bytes += bytes_written
                    total_rows += len(current_rows)
                    logger.info(
                        f"  💾 {table_name} part {file_num}: "
                        f"{len(current_rows):,} rows, "
                        f"{bytes_written / (1024**2):.1f} MB"
                    )
            
            # Update metadata
            self.metadata["exports"][table_name] = {
                "last_export": datetime.utcnow().isoformat(),
                "rows": total_rows,
                "bytes": total_bytes,
                "files": file_num,
            }
            self._save_metadata()
            
            self.stats["tables_exported"] += 1
            self.stats["total_rows"] += total_rows
            self.stats["total_bytes"] += total_bytes
            self.stats["total_files"] += file_num
            
            logger.info(
                f"  ✅ {table_name}: {total_rows:,} rows → "
                f"{file_num} files, {total_bytes / (1024**2):.1f} MB"
            )
            
            return total_rows, total_bytes
            
        except Exception as e:
            error_msg = f"{table_name}: {type(e).__name__}: {str(e)[:200]}"
            logger.error(f"  ❌ {error_msg}")
            self.stats["errors"].append(error_msg)
            return 0, 0
    
    async def _write_csv_file(
        self,
        table_name: str,
        rows: List[Dict],
        headers: List[str],
        file_num: int,
        since: Optional[datetime],
    ) -> int:
        """Write rows to a compressed CSV file. Returns bytes written."""
        now = datetime.utcnow()
        
        # Directory structure: db-exports/{table}/{YYYY}/{MM}/
        dir_path = self.export_path / table_name / now.strftime("%Y") / now.strftime("%m")
        dir_path.mkdir(parents=True, exist_ok=True)
        
        # Filename
        timestamp = now.strftime("%d_%H%M%S")
        suffix = "incremental" if since else "full"
        file_path = dir_path / f"{timestamp}_{table_name}_part{file_num:04d}_{suffix}.csv.gz"
        
        # Build CSV in memory
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            # Convert non-serializable types
            clean_row = {}
            for k, v in row.items():
                if v is None:
                    clean_row[k] = ""
                elif isinstance(v, (datetime,)):
                    clean_row[k] = v.isoformat()
                else:
                    clean_row[k] = str(v)
            writer.writerow(clean_row)
        
        csv_bytes = output.getvalue().encode("utf-8")
        
        # Gzip compress
        compressed = gzip.compress(csv_bytes, compresslevel=6)
        
        # Write to HDD
        with open(file_path, "wb") as f:
            f.write(compressed)
        
        return len(compressed)
    
    async def export_all(self, full: bool = False):
        """Export all tables."""
        mode = "FULL" if full else "INCREMENTAL"
        logger.info("=" * 60)
        logger.info(f"  ROYALEY DB → HDD EXPORT ({mode})")
        logger.info(f"  Target: {self.export_path}")
        logger.info(f"  Tables: {len(EXPORT_TABLES)}")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        from app.core.database import db_manager
        await db_manager.initialize()
        
        for table_def in EXPORT_TABLES:
            table_name = table_def["table"]
            
            # Determine 'since' for incremental
            since = None
            if not full:
                last_export = self.metadata.get("exports", {}).get(table_name, {}).get("last_export")
                if last_export:
                    since = datetime.fromisoformat(last_export)
                else:
                    # First time: export last 30 days
                    since = datetime.utcnow() - timedelta(days=30)
            
            await self.export_table(table_def, since=since, full=full)
        
        elapsed = time.time() - start_time
        
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"  EXPORT COMPLETE ({elapsed:.0f}s)")
        logger.info(f"  Tables:  {self.stats['tables_exported']}")
        logger.info(f"  Rows:    {self.stats['total_rows']:,}")
        logger.info(f"  Files:   {self.stats['total_files']}")
        logger.info(f"  Size:    {self.stats['total_bytes'] / (1024**3):.2f} GB")
        if self.stats["errors"]:
            logger.info(f"  Errors:  {len(self.stats['errors'])}")
            for err in self.stats["errors"]:
                logger.info(f"    ❌ {err}")
        logger.info("=" * 60)
    
    async def show_status(self):
        """Show export status and HDD usage."""
        logger.info("")
        logger.info("=" * 60)
        logger.info("  DB → HDD EXPORT STATUS")
        logger.info(f"  Path: {self.export_path}")
        logger.info("=" * 60)
        
        # Show per-table status
        logger.info(f"  {'Table':<20} {'Last Export':<22} {'Rows':>12} {'Size':>10}")
        logger.info(f"  {'-'*65}")
        
        total_rows = 0
        total_bytes = 0
        
        for table_def in EXPORT_TABLES:
            table_name = table_def["table"]
            export_info = self.metadata.get("exports", {}).get(table_name, {})
            
            last = export_info.get("last_export", "never")
            if last != "never":
                last = last[:19]  # Trim microseconds
            rows = export_info.get("rows", 0)
            size = export_info.get("bytes", 0)
            
            total_rows += rows
            total_bytes += size
            
            size_str = f"{size / (1024**2):.1f} MB" if size > 0 else "—"
            rows_str = f"{rows:,}" if rows > 0 else "—"
            
            logger.info(f"  {table_name:<20} {last:<22} {rows_str:>12} {size_str:>10}")
        
        logger.info(f"  {'-'*65}")
        logger.info(f"  {'TOTAL':<20} {'':22} {total_rows:>12,} {total_bytes / (1024**3):>9.2f} GB")
        
        # Disk scan
        if self.export_path.exists():
            import shutil
            actual_size = sum(
                f.stat().st_size for f in self.export_path.rglob("*") if f.is_file()
            )
            file_count = sum(1 for f in self.export_path.rglob("*") if f.is_file())
            logger.info("")
            logger.info(f"  📁 Actual on disk: {actual_size / (1024**3):.2f} GB ({file_count:,} files)")
            
            try:
                usage = shutil.disk_usage(str(self.export_path))
                logger.info(
                    f"  💿 HDD: {usage.used / (1024**4):.2f} TB used / "
                    f"{usage.total / (1024**4):.1f} TB total"
                )
            except Exception:
                pass
        
        logger.info("=" * 60)


# =============================================================================
# MAIN
# =============================================================================

async def main_async(args):
    exporter = DBExporter()
    
    if args.status:
        await exporter.show_status()
    elif args.table:
        # Export specific table
        from app.core.database import db_manager
        await db_manager.initialize()
        
        table_def = next((t for t in EXPORT_TABLES if t["table"] == args.table), None)
        if not table_def:
            logger.error(f"Unknown table: {args.table}")
            logger.info(f"Available: {[t['table'] for t in EXPORT_TABLES]}")
            return
        
        since = None
        if not args.full:
            since = datetime.utcnow() - timedelta(days=args.days)
        
        await exporter.export_table(table_def, since=since, full=args.full)
    else:
        await exporter.export_all(full=args.full)


def main():
    parser = argparse.ArgumentParser(description="ROYALEY DB → HDD Export")
    parser.add_argument("--full", action="store_true", help="Full export (all data)")
    parser.add_argument("--incremental", action="store_true", help="Incremental export (default)")
    parser.add_argument("--table", "-t", help="Export specific table")
    parser.add_argument("--days", "-d", type=int, default=30, help="Days back for incremental (default: 30)")
    parser.add_argument("--status", action="store_true", help="Show export status")
    
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()