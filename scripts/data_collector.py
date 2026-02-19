#!/usr/bin/env python3
"""
ROYALEY - Automated Data Collection Service
=============================================

Runs all 27 data collectors on smart schedules, archiving everything to HDD (16TB).
Designed to run as a Docker service alongside the main API and worker.

Schedules:
  FAST (every 30 min):   odds_api, pinnacle, espn (injuries)
  MEDIUM (every 2 hrs):  weather, weatherstack, sportsdb, espn, nhl_api
  SLOW (every 6 hrs):    R-data packages, tennis, basketball_ref, etc.
  DAILY (once/day):      kaggle, polymarket, kalshi, realgm, nextgenstats
  WEEKLY (Sunday 3AM):   Full historical backfill

Usage:
    # Inside Docker:
    python scripts/data_collector.py                    # Start scheduled collection
    python scripts/data_collector.py --initial          # Run initial full import first, then schedule
    python scripts/data_collector.py --status            # Show HDD storage stats
    python scripts/data_collector.py --test              # Test one round of all collectors

    # From host:
    docker exec royaley_data_collector python scripts/data_collector.py --status
"""

import asyncio
import argparse
import logging
import os
import signal
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("/app/logs/data_collector.log", mode="a"),
    ],
)
logger = logging.getLogger("royaley.data_collector")

shutdown_flag = False


def signal_handler(signum, frame):
    global shutdown_flag
    logger.info("Shutdown signal received, finishing current cycle...")
    shutdown_flag = True


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# =============================================================================
# COLLECTION SCHEDULES
# =============================================================================

# Each tuple: (source_key, description, interval_seconds)
# source_key must match IMPORT_MAP in master_import.py

FAST_SOURCES = [
    # Every 30 minutes - live/real-time data
    ("odds_api", "TheOddsAPI (live odds)", 1800),
    ("pinnacle", "Pinnacle (CLV benchmark)", 1800),
    ("injuries", "ESPN Injuries", 1800),
]

MEDIUM_SOURCES = [
    # Every 2 hours - game data, weather
    ("espn", "ESPN (games, teams)", 7200),
    ("weather", "OpenWeatherMap", 7200),
    ("sportsdb", "TheSportsDB (games, scores)", 7200),
    ("nhl_api", "NHL Official API", 7200),
    ("balldontlie", "BallDontLie (multi-sport)", 7200),
]

SLOW_SOURCES = [
    # Every 6 hours - R data packages, stats
    ("nflfastr", "nflfastR (NFL)", 21600),
    ("cfbfastr", "cfbfastR (NCAAF)", 21600),
    ("baseballr", "baseballR (MLB)", 21600),
    ("hockeyr", "hockeyR (NHL)", 21600),
    ("wehoop", "wehoop (WNBA)", 21600),
    ("hoopr", "hoopR (NBA/NCAAB)", 21600),
    ("cfl", "CFL Official API", 21600),
    ("tennis_abstract", "Tennis Abstract (ATP/WTA)", 21600),
    ("cfbd", "College Football Data", 21600),
    ("matchstat", "Matchstat Tennis", 21600),
    ("weatherstack", "Weatherstack (backup)", 21600),
]

DAILY_SOURCES = [
    # Once per day - bulk/market data
    ("action_network", "Action Network (public betting)", 86400),
    ("basketball_ref", "Basketball Reference", 86400),
    ("sportsipy", "Sportsipy (multi-sport)", 86400),
    ("realgm", "RealGM (NBA salaries)", 86400),
    ("nextgenstats", "NFL Next Gen Stats", 86400),
    ("kaggle", "Kaggle Datasets", 86400),
    ("polymarket", "Polymarket (prediction markets)", 86400),
    ("kalshi", "Kalshi (prediction markets)", 86400),
    ("closing_lines", "Pinnacle Closing Lines", 86400),
]

# Historical backfill sources (run weekly or on initial import)
HISTORICAL_SOURCES = [
    "pinnacle_history", "espn_history", "odds_api_history",
    "sportsdb_history", "nflfastr_history", "cfbfastr_history",
    "baseballr_history", "hockeyr_history", "wehoop_history",
    "hoopr_history", "cfl_history", "tennis_abstract_history",
    "nhl_api_history", "weather_history", "cfbd_history",
    "matchstat_history", "kaggle_history", "kalshi_history",
]

ALL_SCHEDULED = FAST_SOURCES + MEDIUM_SOURCES + SLOW_SOURCES + DAILY_SOURCES


# =============================================================================
# IMPORT EXECUTION
# =============================================================================

async def run_single_import(source_key: str, sports: List[str] = None) -> Tuple[bool, int, str]:
    """
    Run a single import source and return (success, records, error_msg).
    Uses the existing master_import.py infrastructure.
    """
    try:
        # Import the IMPORT_MAP lazily to avoid circular imports at module load
        from scripts.master_import import IMPORT_MAP
        
        func = IMPORT_MAP.get(source_key)
        if not func:
            return False, 0, f"Unknown source: {source_key}"
        
        # Call with appropriate params
        if sports:
            result = await func(sports=sports)
        else:
            result = await func()
        
        return result.success, result.records, "; ".join(result.errors) if result.errors else ""
        
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)[:200]}"
        logger.error(f"Import {source_key} failed: {error_msg}")
        return False, 0, error_msg


# =============================================================================
# SCHEDULER LOOP
# =============================================================================

class DataCollectionScheduler:
    """Manages scheduled data collection from all 27 sources."""
    
    def __init__(self):
        self.last_run: Dict[str, float] = {}  # source_key → timestamp
        self.run_counts: Dict[str, int] = {}
        self.total_records: Dict[str, int] = {}
        self.error_counts: Dict[str, int] = {}
        self.start_time = time.time()
    
    def is_due(self, source_key: str, interval_seconds: int) -> bool:
        """Check if a source is due for collection."""
        last = self.last_run.get(source_key, 0)
        return (time.time() - last) >= interval_seconds
    
    def mark_complete(self, source_key: str, records: int, success: bool):
        """Mark a source as completed."""
        self.last_run[source_key] = time.time()
        self.run_counts[source_key] = self.run_counts.get(source_key, 0) + 1
        if success:
            self.total_records[source_key] = self.total_records.get(source_key, 0) + records
        else:
            self.error_counts[source_key] = self.error_counts.get(source_key, 0) + 1
    
    async def run_due_collections(self) -> Dict[str, any]:
        """Run all sources that are due for collection."""
        results = {}
        
        for source_key, description, interval in ALL_SCHEDULED:
            if shutdown_flag:
                break
            
            if not self.is_due(source_key, interval):
                continue
            
            logger.info(f"📡 Collecting: {description} [{source_key}]")
            start = time.time()
            
            success, records, error = await run_single_import(source_key)
            elapsed = time.time() - start
            
            self.mark_complete(source_key, records, success)
            
            if success:
                logger.info(f"  ✅ {source_key}: {records} records in {elapsed:.1f}s")
            else:
                logger.warning(f"  ❌ {source_key}: {error[:100]} ({elapsed:.1f}s)")
            
            results[source_key] = {"success": success, "records": records, "elapsed": elapsed}
            
            # Small delay between sources to avoid overwhelming APIs
            await asyncio.sleep(2)
        
        return results
    
    async def run_historical_backfill(self, seasons: int = 10):
        """Run full historical import (initial or weekly)."""
        logger.info("=" * 60)
        logger.info("🏆 STARTING HISTORICAL BACKFILL")
        logger.info(f"   Seasons: {seasons}")
        logger.info("=" * 60)
        
        for source_key in HISTORICAL_SOURCES:
            if shutdown_flag:
                break
            
            logger.info(f"📜 Historical: {source_key}")
            start = time.time()
            
            success, records, error = await run_single_import(source_key)
            elapsed = time.time() - start
            
            if success:
                logger.info(f"  ✅ {source_key}: {records} records in {elapsed:.1f}s")
            else:
                logger.warning(f"  ❌ {source_key}: {error[:100]}")
            
            await asyncio.sleep(5)  # Longer delay for historical pulls
    
    def print_status(self):
        """Print current collection status."""
        uptime = time.time() - self.start_time
        hours = uptime / 3600
        
        logger.info("")
        logger.info("=" * 60)
        logger.info("  DATA COLLECTION STATUS")
        logger.info("=" * 60)
        logger.info(f"  Uptime: {hours:.1f} hours")
        logger.info(f"  Sources: {len(ALL_SCHEDULED)}")
        logger.info("")
        logger.info(f"  {'Source':<25} {'Runs':>5} {'Records':>10} {'Errors':>6} {'Last Run':<20}")
        logger.info(f"  {'-'*70}")
        
        for source_key, desc, interval in ALL_SCHEDULED:
            runs = self.run_counts.get(source_key, 0)
            records = self.total_records.get(source_key, 0)
            errors = self.error_counts.get(source_key, 0)
            last = self.last_run.get(source_key, 0)
            last_str = datetime.fromtimestamp(last).strftime("%m/%d %H:%M") if last else "never"
            
            logger.info(f"  {source_key:<25} {runs:>5} {records:>10,} {errors:>6} {last_str:<20}")
        
        logger.info("=" * 60)


async def run_scheduler():
    """Main scheduler loop - runs continuously."""
    scheduler = DataCollectionScheduler()
    
    logger.info("=" * 60)
    logger.info("🚀 ROYALEY DATA COLLECTION SERVICE STARTED")
    logger.info(f"   Sources: {len(ALL_SCHEDULED)} collectors")
    logger.info(f"   Fast cycle: {len(FAST_SOURCES)} sources every 30 min")
    logger.info(f"   Medium cycle: {len(MEDIUM_SOURCES)} sources every 2 hrs")
    logger.info(f"   Slow cycle: {len(SLOW_SOURCES)} sources every 6 hrs")
    logger.info(f"   Daily cycle: {len(DAILY_SOURCES)} sources once/day")
    logger.info(f"   Archive: /app/raw-data (HDD 16TB)")
    logger.info("=" * 60)
    
    # Print HDD status
    await print_hdd_status()
    
    cycle = 0
    last_weekly = 0
    last_db_export = 0
    
    while not shutdown_flag:
        cycle += 1
        logger.info(f"\n--- Collection Cycle #{cycle} ({datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}) ---")
        
        # Run all due collections
        results = await scheduler.run_due_collections()
        
        if results:
            successful = sum(1 for r in results.values() if r["success"])
            total_recs = sum(r["records"] for r in results.values())
            logger.info(f"  📊 Cycle #{cycle}: {successful}/{len(results)} succeeded, {total_recs:,} records")
        
        now = datetime.utcnow()
        
        # Daily DB → HDD export (2 AM UTC)
        if now.hour == 2 and (time.time() - last_db_export) > 43200:
            logger.info("💾 Daily DB → HDD export starting...")
            try:
                from scripts.db_to_hdd_export import DBExporter
                exporter = DBExporter()
                await exporter.export_all(full=False)
            except Exception as e:
                logger.error(f"DB export failed: {e}")
            last_db_export = time.time()
        
        # Weekly historical backfill (Sunday 3 AM UTC)
        if now.weekday() == 6 and now.hour == 3 and (time.time() - last_weekly) > 86400:
            logger.info("📅 Weekly historical backfill starting...")
            await scheduler.run_historical_backfill(seasons=2)  # Last 2 seasons only for weekly
            last_weekly = time.time()
        
        # Status report every 10 cycles
        if cycle % 10 == 0:
            scheduler.print_status()
            await print_hdd_status()
        
        # Sleep until next check (60 seconds)
        for _ in range(60):
            if shutdown_flag:
                break
            await asyncio.sleep(1)
    
    logger.info("Data collection service stopped.")
    scheduler.print_status()


# =============================================================================
# INITIAL IMPORT (First run - fills HDD with historical data)
# =============================================================================

async def run_initial_import():
    """
    Run the comprehensive initial import.
    This pulls 10 years of historical data from all sources.
    Expected to take 2-6 hours depending on API limits.
    """
    logger.info("=" * 60)
    logger.info("🏆 INITIAL FULL DATA IMPORT")
    logger.info("   This will pull 10 years of historical data")
    logger.info("   Expected: 2-6 hours, 100GB+ of raw data")
    logger.info("=" * 60)
    
    # Step 1: Current data first (establishes teams, foreign keys)
    current_sources = [
        "espn", "odds_api", "pinnacle", "weather", "sportsdb",
        "balldontlie", "nhl_api", "cfbd",
    ]
    
    logger.info("\n📡 Phase 1: Current data (teams, games, odds)...")
    for source in current_sources:
        if shutdown_flag:
            break
        logger.info(f"  Importing {source}...")
        success, records, error = await run_single_import(source)
        status = f"✅ {records} records" if success else f"❌ {error[:80]}"
        logger.info(f"  {source}: {status}")
        await asyncio.sleep(3)
    
    # Step 2: Historical data (10 years per sport)
    logger.info("\n📜 Phase 2: Historical data (10 years)...")
    historical_order = [
        "nflfastr_history", "cfbfastr_history", "baseballr_history",
        "hockeyr_history", "hoopr_history", "wehoop_history",
        "cfl_history", "tennis_abstract_history",
        "sportsdb_history", "nhl_api_history",
        "pinnacle_history", "odds_api_history",
        "weather_history", "cfbd_history",
        "matchstat_history", "kaggle_history",
    ]
    
    for source in historical_order:
        if shutdown_flag:
            break
        logger.info(f"  Importing {source}...")
        success, records, error = await run_single_import(source)
        status = f"✅ {records} records" if success else f"❌ {error[:80]}"
        logger.info(f"  {source}: {status}")
        await asyncio.sleep(5)
    
    # Step 3: Players, injuries, specialized
    logger.info("\n👤 Phase 3: Players, injuries, specialized data...")
    player_sources = [
        "players", "injuries", "nfl_players", "mlb_players",
        "nhl_players", "nba_players", "wnba_players",
        "realgm", "nextgenstats", "kaggle",
        "polymarket", "kalshi",
    ]
    
    for source in player_sources:
        if shutdown_flag:
            break
        logger.info(f"  Importing {source}...")
        success, records, error = await run_single_import(source)
        status = f"✅ {records} records" if success else f"❌ {error[:80]}"
        logger.info(f"  {source}: {status}")
        await asyncio.sleep(3)
    
    logger.info("\n" + "=" * 60)
    logger.info("📊 INITIAL IMPORT COMPLETE - Starting DB → HDD export...")
    logger.info("=" * 60)
    
    # Phase 4: Export all DB data to HDD as compressed CSV
    try:
        from scripts.db_to_hdd_export import DBExporter
        exporter = DBExporter()
        await exporter.export_all(full=True)
    except Exception as e:
        logger.error(f"DB export failed (non-fatal): {e}")
    
    await print_hdd_status()


# =============================================================================
# HDD STATUS
# =============================================================================

async def print_hdd_status():
    """Print HDD storage status."""
    try:
        from app.services.data.raw_data_archiver import get_archiver
        archiver = get_archiver()
        report = archiver.print_storage_report()
        logger.info(report)
    except Exception as e:
        logger.warning(f"Could not get HDD status: {e}")
        # Fallback: basic disk check
        try:
            import shutil
            usage = shutil.disk_usage("/app/raw-data")
            logger.info(f"  HDD: {usage.used / (1024**4):.2f} TB used / {usage.total / (1024**4):.1f} TB total "
                        f"({usage.free / (1024**4):.1f} TB free)")
        except Exception:
            logger.warning("  Cannot read /app/raw-data disk info")


# =============================================================================
# TEST MODE
# =============================================================================

async def run_test():
    """Run one collection from each source to verify connectivity."""
    logger.info("🧪 TEST MODE: Running one collection per source...")
    
    results = []
    for source_key, desc, _ in ALL_SCHEDULED:
        logger.info(f"  Testing {source_key}...")
        success, records, error = await run_single_import(source_key)
        results.append((source_key, success, records, error))
        
        if success:
            logger.info(f"    ✅ {records} records")
        else:
            logger.warning(f"    ❌ {error[:80]}")
        
        await asyncio.sleep(2)
    
    # Summary
    ok = sum(1 for _, s, _, _ in results if s)
    logger.info(f"\n📊 Test Results: {ok}/{len(results)} sources working")
    
    for source, success, records, error in results:
        icon = "✅" if success else "❌"
        logger.info(f"  {icon} {source:<25} {records:>6} records  {error[:40] if error else ''}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="ROYALEY Data Collection Service")
    
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--initial", action="store_true", help="Run initial full import then start scheduler")
    mode.add_argument("--status", action="store_true", help="Show HDD storage stats")
    mode.add_argument("--test", action="store_true", help="Test one round of all collectors")
    mode.add_argument("--historical", action="store_true", help="Run historical backfill only")
    mode.add_argument("--db-export", action="store_true", help="Export all DB tables to HDD")
    mode.add_argument("--db-export-full", action="store_true", help="Full DB export (all data, not incremental)")
    
    args = parser.parse_args()
    
    # Ensure archive directories exist
    try:
        from app.services.data.raw_data_archiver import get_archiver
        archiver = get_archiver()
        if not archiver.enabled:
            logger.error("❌ Raw data archiver is DISABLED. Check /app/raw-data permissions.")
            sys.exit(1)
        logger.info(f"✅ Archive enabled at {archiver.base_path}")
    except Exception as e:
        logger.error(f"❌ Cannot initialize archiver: {e}")
        sys.exit(1)
    
    if args.status:
        asyncio.run(print_hdd_status())
    elif args.test:
        asyncio.run(run_test())
    elif args.historical:
        scheduler = DataCollectionScheduler()
        asyncio.run(scheduler.run_historical_backfill(seasons=10))
    elif args.db_export or args.db_export_full:
        async def do_export():
            from scripts.db_to_hdd_export import DBExporter
            exporter = DBExporter()
            await exporter.export_all(full=args.db_export_full)
        asyncio.run(do_export())
    elif args.initial:
        async def initial_then_schedule():
            await run_initial_import()
            logger.info("\n🔄 Initial import done. Starting scheduled collection...\n")
            await run_scheduler()
        asyncio.run(initial_then_schedule())
    else:
        # Default: run scheduled collection
        asyncio.run(run_scheduler())


if __name__ == "__main__":
    main()