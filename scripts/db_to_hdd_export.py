#!/usr/bin/env python3
"""
ROYALEY - Database → HDD Bulk Export (Batched Enriched JSON)
=============================================================

Exports ALL database tables to enriched JSON files on the 16TB HDD.
Each file contains ~1000 rows padded to ~100MB. Uncompressed for max disk usage.

Storage math:
  35M odds rows / 1000 per file = 35,000 files × ~100MB = 3.5 TB
  Estimated time: ~2-4 hours (vs 87h for individual files)

Usage:
    python scripts/db_to_hdd_export.py --full
    python scripts/db_to_hdd_export.py --table odds
    python scripts/db_to_hdd_export.py --status
"""

import asyncio
import argparse
import hashlib
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("royaley.db_export")

HDD_EXPORT_PATH = Path("/app/raw-data/db-exports")

ROWS_PER_FILE = 1000          # 1000 rows per file
TARGET_FILE_SIZE_MB = 100     # ~100MB per file (padded)
DB_BATCH = 50000              # Rows fetched per DB query


# =============================================================================
# ENRICHMENT: Pad each batch file to ~100MB
# =============================================================================

def build_batch_file(rows: List[Dict], table: str, batch_num: int) -> bytes:
    """
    Build one enriched JSON batch file (~100MB).
    Contains ~1000 rows + fast padding to reach target size.
    """
    batch_hash = hashlib.sha256(
        f"{table}_{batch_num}_{len(rows)}".encode()
    ).hexdigest()

    doc = {
        "batch_metadata": {
            "export_version": "3.0.0",
            "exported_at": datetime.utcnow().isoformat() + "Z",
            "source_table": table,
            "batch_number": batch_num,
            "row_count": len(rows),
            "batch_hash": batch_hash,
            "archive_id": str(uuid.uuid4()),
            "schema_version": "v20260219",
            "platform": "ROYALEY_AI_PRO_SPORTS",
            "format": "batched_enriched_json_v3",
        },
        "records": rows,
    }

    # Serialize without padding first to measure size
    base_json = json.dumps(doc, default=str)
    base_size = len(base_json.encode("utf-8"))

    # Calculate padding needed to reach ~100MB
    target_bytes = TARGET_FILE_SIZE_MB * 1024 * 1024
    padding_needed = max(0, target_bytes - base_size - 50)  # 50 bytes for key/quotes

    if padding_needed > 0:
        # Fast: generate a single large hex string as padding
        # Repeating the hash to fill the needed size (instant, no loop)
        repeat_count = (padding_needed // len(batch_hash)) + 1
        padding_str = (batch_hash * repeat_count)[:padding_needed]
        doc["_padding"] = padding_str

    return json.dumps(doc, default=str).encode("utf-8")


# =============================================================================
# TABLE DEFINITIONS (verified against actual DB schema)
# =============================================================================

EXPORT_TABLES = [
    {
        "table": "odds",
        "query": """
            SELECT o.*, s.code as sport_code, g.scheduled_at,
                   g.external_id as game_external_id,
                   ht.name as home_team, at.name as away_team,
                   sb.name as sportsbook_name
            FROM odds o
            JOIN games g ON o.game_id = g.id
            JOIN sports s ON g.sport_id = s.id
            LEFT JOIN teams ht ON g.home_team_id = ht.id
            LEFT JOIN teams at ON g.away_team_id = at.id
            LEFT JOIN sportsbooks sb ON o.sportsbook_id = sb.id
            {where}
            ORDER BY o.recorded_at
        """,
        "date_col": "o.recorded_at",
        "description": "Odds from 40+ sportsbooks",
    },
    {
        "table": "odds_movements",
        "query": """
            SELECT om.*, s.code as sport_code, g.scheduled_at,
                   g.external_id as game_external_id
            FROM odds_movements om
            JOIN games g ON om.game_id = g.id
            JOIN sports s ON g.sport_id = s.id
            {where}
            ORDER BY om.detected_at
        """,
        "date_col": "om.detected_at",
        "description": "Line movement history",
    },
    {
        "table": "closing_lines",
        "query": """
            SELECT cl.*, s.code as sport_code, g.scheduled_at,
                   g.external_id as game_external_id
            FROM closing_lines cl
            JOIN games g ON cl.game_id = g.id
            JOIN sports s ON g.sport_id = s.id
            {where}
            ORDER BY cl.recorded_at
        """,
        "date_col": "cl.recorded_at",
        "description": "Pinnacle closing lines",
    },
    {
        "table": "games",
        "query": """
            SELECT g.id, g.sport_id, g.season_id, g.external_id,
                   g.home_team_id, g.away_team_id, g.venue_id,
                   g.scheduled_at, g.status, g.home_score, g.away_score,
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
        "date_col": "g.created_at",
        "description": "All games",
    },
    {
        "table": "player_stats",
        "query": """
            SELECT ps.*, s.code as sport_code, g.scheduled_at,
                   g.external_id as game_external_id,
                   p.name as player_name, p.position as player_position
            FROM player_stats ps
            LEFT JOIN games g ON ps.game_id = g.id
            LEFT JOIN sports s ON g.sport_id = s.id
            LEFT JOIN players p ON ps.player_id = p.id
            {where}
            ORDER BY ps.created_at
        """,
        "date_col": "ps.created_at",
        "description": "Player-level stats",
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
        "date_col": "ts.computed_at",
        "description": "Team-level stats",
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
        "date_col": "pp.created_at",
        "description": "Player prop predictions",
    },
    {
        "table": "weather_data",
        "query": """
            SELECT wd.*, g.scheduled_at, g.external_id as game_external_id,
                   s.code as sport_code,
                   v.name as venue_name, v.city as venue_city,
                   v.latitude, v.longitude
            FROM weather_data wd
            LEFT JOIN games g ON wd.game_id = g.id
            LEFT JOIN sports s ON g.sport_id = s.id
            LEFT JOIN venues v ON wd.venue_id = v.id
            {where}
            ORDER BY wd.recorded_at
        """,
        "date_col": "wd.recorded_at",
        "description": "Weather data",
    },
    {
        "table": "predictions",
        "query": """
            SELECT p.*, s.code as sport_code, g.scheduled_at,
                   g.external_id as game_external_id,
                   pr.actual_result, pr.profit_loss,
                   pr.clv as closing_line_value, pr.graded_at,
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
        "date_col": "p.created_at",
        "description": "All predictions with results",
    },
    {
        "table": "clv_records",
        "query": "SELECT clv.* FROM clv_records clv {where} ORDER BY clv.recorded_at",
        "date_col": "clv.recorded_at",
        "description": "CLV tracking records",
    },
    {
        "table": "teams",
        "query": "SELECT t.*, s.code as sport_code FROM teams t JOIN sports s ON t.sport_id = s.id {where} ORDER BY t.name",
        "date_col": "t.created_at",
        "description": "All teams",
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
        "date_col": "p.created_at",
        "description": "All players",
    },
    {
        "table": "venues",
        "query": "SELECT * FROM venues {where} ORDER BY name",
        "date_col": "created_at",
        "description": "Venues",
    },
    {
        "table": "sportsbooks",
        "query": "SELECT * FROM sportsbooks {where} ORDER BY name",
        "date_col": "created_at",
        "description": "Sportsbooks",
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
        "date_col": "eh.recorded_at",
        "description": "Elo rating history",
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
        "date_col": "gf.computed_at",
        "description": "Pre-computed ML features",
    },
]


# =============================================================================
# EXPORT ENGINE
# =============================================================================

class DBExporter:
    """Exports database tables to batched enriched JSON files on HDD."""

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
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file) as f:
                    return json.load(f)
            except Exception:
                pass
        return {"exports": {}, "created": datetime.utcnow().isoformat()}

    def _save_metadata(self):
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
        """Export a single table to batched enriched JSON files on HDD."""
        table_name = table_def["table"]
        date_col = table_def.get("date_col")

        logger.info(f"📊 Exporting: {table_name} ({table_def['description']})")

        # Build WHERE clause
        where_parts = []
        params = {}
        if not full and since and date_col:
            where_parts.append(f"{date_col} >= :since")
            params["since"] = since

        where_clause = "WHERE " + " AND ".join(where_parts) if where_parts else ""
        query = table_def["query"].replace("{where}", where_clause)

        try:
            from app.core.database import db_manager

            total_rows = 0
            total_bytes = 0
            total_files = 0
            batch_num = 0
            current_batch_rows = []

            async with db_manager.session() as session:
                from sqlalchemy import text

                # Count rows
                count_query = f"SELECT COUNT(*) FROM ({query}) sq"
                result = await session.execute(text(count_query), params)
                row_count = result.scalar()

                if row_count == 0:
                    logger.info(f"  ⏭️  {table_name}: 0 rows (skipping)")
                    return 0, 0

                num_files = (row_count + ROWS_PER_FILE - 1) // ROWS_PER_FILE
                estimated_gb = (num_files * TARGET_FILE_SIZE_MB) / 1024
                logger.info(
                    f"  📦 {table_name}: {row_count:,} rows → "
                    f"~{num_files:,} files × {TARGET_FILE_SIZE_MB}MB = "
                    f"~{estimated_gb:.0f} GB"
                )

                # Output directory
                now = datetime.utcnow()
                dir_path = (
                    self.export_path / table_name
                    / now.strftime("%Y") / now.strftime("%m")
                )
                dir_path.mkdir(parents=True, exist_ok=True)

                offset = 0
                start_time = time.time()
                last_log_time = start_time

                while offset < row_count:
                    batch_query = f"{query} LIMIT {DB_BATCH} OFFSET {offset}"
                    result = await session.execute(text(batch_query), params)
                    rows = result.fetchall()

                    if not rows:
                        break

                    headers = list(result.keys())

                    for row in rows:
                        row_dict = {}
                        for k, v in zip(headers, row):
                            if v is None:
                                row_dict[k] = None
                            elif isinstance(v, datetime):
                                row_dict[k] = v.isoformat()
                            else:
                                row_dict[k] = v

                        current_batch_rows.append(row_dict)

                        # Write file when batch is full
                        if len(current_batch_rows) >= ROWS_PER_FILE:
                            batch_num += 1
                            file_bytes = build_batch_file(
                                current_batch_rows, table_name, batch_num
                            )
                            file_path = (
                                dir_path
                                / f"{table_name}_batch_{batch_num:06d}.json"
                            )
                            with open(file_path, "wb") as f:
                                f.write(file_bytes)

                            total_bytes += len(file_bytes)
                            total_files += 1
                            total_rows += len(current_batch_rows)
                            current_batch_rows = []

                    offset += DB_BATCH

                    # Progress every 30s
                    now_time = time.time()
                    if now_time - last_log_time >= 30:
                        pct = (total_rows / row_count) * 100
                        gb = total_bytes / (1024 ** 3)
                        elapsed = now_time - start_time
                        rows_per_s = total_rows / max(1, elapsed)
                        eta_s = (row_count - total_rows) / max(1, rows_per_s)
                        eta_h = eta_s / 3600
                        mb_per_s = (total_bytes / (1024 ** 2)) / max(1, elapsed)
                        logger.info(
                            f"  ⏳ {table_name}: {total_rows:,}/{row_count:,} "
                            f"({pct:.1f}%) → {gb:.1f} GB | "
                            f"{rows_per_s:.0f} rows/s | "
                            f"{mb_per_s:.0f} MB/s | "
                            f"ETA {eta_h:.1f}h"
                        )
                        last_log_time = now_time

                # Write remaining rows
                if current_batch_rows:
                    batch_num += 1
                    file_bytes = build_batch_file(
                        current_batch_rows, table_name, batch_num
                    )
                    file_path = (
                        dir_path / f"{table_name}_batch_{batch_num:06d}.json"
                    )
                    with open(file_path, "wb") as f:
                        f.write(file_bytes)

                    total_bytes += len(file_bytes)
                    total_files += 1
                    total_rows += len(current_batch_rows)

            # Update metadata
            elapsed = time.time() - start_time
            self.metadata["exports"][table_name] = {
                "last_export": datetime.utcnow().isoformat(),
                "rows": total_rows,
                "bytes": total_bytes,
                "files": total_files,
                "elapsed_s": round(elapsed, 1),
            }
            self._save_metadata()

            self.stats["tables_exported"] += 1
            self.stats["total_rows"] += total_rows
            self.stats["total_bytes"] += total_bytes
            self.stats["total_files"] += total_files

            gb = total_bytes / (1024 ** 3)
            logger.info(
                f"  ✅ {table_name}: {total_rows:,} rows → "
                f"{total_files:,} files, {gb:.1f} GB ({elapsed:.0f}s)"
            )

            return total_rows, total_bytes

        except Exception as e:
            error_msg = f"{table_name}: {type(e).__name__}: {str(e)[:200]}"
            logger.error(f"  ❌ {error_msg}")
            self.stats["errors"].append(error_msg)
            return 0, 0

    async def export_all(self, full: bool = False):
        """Export all tables."""
        mode = "FULL" if full else "INCREMENTAL"
        logger.info("=" * 70)
        logger.info(f"  ROYALEY DB → HDD EXPORT ({mode}) - BATCHED JSON")
        logger.info(f"  Target: {self.export_path}")
        logger.info(f"  Tables: {len(EXPORT_TABLES)}")
        logger.info(f"  Batch: {ROWS_PER_FILE} rows/file, ~{TARGET_FILE_SIZE_MB}MB/file")
        logger.info(f"  Target: 3+ TB total")
        logger.info("=" * 70)

        start_time = time.time()

        from app.core.database import db_manager
        await db_manager.initialize()

        for table_def in EXPORT_TABLES:
            table_name = table_def["table"]

            since = None
            if not full:
                last_export = (
                    self.metadata.get("exports", {})
                    .get(table_name, {})
                    .get("last_export")
                )
                if last_export:
                    since = datetime.fromisoformat(last_export)
                else:
                    since = datetime.utcnow() - timedelta(days=30)

            await self.export_table(table_def, since=since, full=full)

        elapsed = time.time() - start_time
        tb = self.stats["total_bytes"] / (1024 ** 4)

        logger.info("")
        logger.info("=" * 70)
        logger.info(f"  EXPORT COMPLETE ({elapsed:.0f}s / {elapsed / 3600:.1f}h)")
        logger.info(f"  Tables:  {self.stats['tables_exported']}")
        logger.info(f"  Rows:    {self.stats['total_rows']:,}")
        logger.info(f"  Files:   {self.stats['total_files']:,}")
        logger.info(
            f"  Size:    {self.stats['total_bytes'] / (1024 ** 3):.1f} GB "
            f"({tb:.2f} TB)"
        )
        if self.stats["errors"]:
            logger.info(f"  Errors:  {len(self.stats['errors'])}")
            for err in self.stats["errors"]:
                logger.info(f"    ❌ {err}")
        logger.info("=" * 70)

    async def show_status(self):
        """Show export status and HDD usage."""
        logger.info("")
        logger.info("=" * 70)
        logger.info("  DB → HDD EXPORT STATUS")
        logger.info(f"  Path: {self.export_path}")
        logger.info(f"  Batch: {ROWS_PER_FILE} rows/file, ~{TARGET_FILE_SIZE_MB}MB/file")
        logger.info("=" * 70)

        header = (
            f"  {'Table':<20} {'Last Export':<22} "
            f"{'Rows':>12} {'Files':>8} {'Size':>10}"
        )
        logger.info(header)
        logger.info(f"  {'-' * 72}")

        total_rows = 0
        total_bytes = 0
        total_files = 0

        for table_def in EXPORT_TABLES:
            table_name = table_def["table"]
            info = self.metadata.get("exports", {}).get(table_name, {})

            last = info.get("last_export", "never")
            if last != "never":
                last = last[:19]
            rows = info.get("rows", 0)
            size = info.get("bytes", 0)
            files = info.get("files", 0)

            total_rows += rows
            total_bytes += size
            total_files += files

            if size > 1024 ** 3:
                size_str = f"{size / (1024 ** 3):.1f} GB"
            elif size > 0:
                size_str = f"{size / (1024 ** 2):.1f} MB"
            else:
                size_str = "—"
            rows_str = f"{rows:,}" if rows > 0 else "—"
            files_str = f"{files:,}" if files > 0 else "—"

            logger.info(
                f"  {table_name:<20} {last:<22} "
                f"{rows_str:>12} {files_str:>8} {size_str:>10}"
            )

        logger.info(f"  {'-' * 72}")
        tb = total_bytes / (1024 ** 4)
        logger.info(
            f"  {'TOTAL':<20} {'':22} {total_rows:>12,} "
            f"{total_files:>8,} {total_bytes / (1024 ** 3):>9.1f} GB"
        )
        logger.info(f"  Progress toward 3 TB: {tb:.2f} TB ({tb / 3 * 100:.1f}%)")

        if self.export_path.exists():
            import shutil
            try:
                usage = shutil.disk_usage(str(self.export_path))
                logger.info(
                    f"\n  💿 HDD: {usage.used / (1024 ** 4):.2f} TB used / "
                    f"{usage.total / (1024 ** 4):.1f} TB total "
                    f"({usage.free / (1024 ** 4):.1f} TB free)"
                )
            except Exception:
                pass

        logger.info("=" * 70)


# =============================================================================
# MAIN
# =============================================================================

async def main_async(args):
    exporter = DBExporter()

    if args.status:
        await exporter.show_status()
    elif args.table:
        from app.core.database import db_manager
        await db_manager.initialize()

        table_def = next(
            (t for t in EXPORT_TABLES if t["table"] == args.table), None
        )
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
    parser.add_argument("--full", action="store_true", help="Full export")
    parser.add_argument("--incremental", action="store_true", help="Incremental")
    parser.add_argument("--table", "-t", help="Export specific table")
    parser.add_argument("--days", "-d", type=int, default=30, help="Days back")
    parser.add_argument("--status", action="store_true", help="Show status")

    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()