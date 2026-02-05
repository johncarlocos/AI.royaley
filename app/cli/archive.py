"""
ROYALEY - Archive Management CLI
Commands for managing the raw data archive on HDD (16TB).

Usage (inside Docker):
    python -m app.cli.archive stats          # Show storage stats
    python -m app.cli.archive report         # Full storage report
    python -m app.cli.archive list           # List recent archives
    python -m app.cli.archive export-db      # Export DB tables to HDD as CSV
    python -m app.cli.archive verify         # Verify archive integrity
"""

import asyncio
import argparse
import json
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def get_archiver():
    """Import and return archiver instance."""
    from app.services.data.raw_data_archiver import get_archiver as _get
    return _get()


async def cmd_stats():
    """Show archive storage statistics."""
    archiver = get_archiver()
    stats = archiver.get_storage_stats(force_refresh=True)
    
    print(f"\n📊 ROYALEY Archive Stats")
    print(f"{'='*50}")
    print(f"Total Size:    {stats['total_size_gb']:.2f} GB ({stats['total_size_tb']:.4f} TB)")
    print(f"Total Files:   {stats['total_files']:,}")
    print(f"Target:        {stats['target_size_tb']:.1f} TB")
    print(f"Progress:      {stats['progress_pct']:.2f}%")
    
    disk = stats.get("disk_info", {})
    if "total_tb" in disk:
        print(f"\n💿 HDD: {disk['used_tb']:.1f} / {disk['total_tb']:.1f} TB "
              f"({disk['free_tb']:.1f} TB free)")


async def cmd_report():
    """Show full storage report."""
    archiver = get_archiver()
    report = archiver.print_storage_report()
    print(report)


async def cmd_list(category=None, source=None, sport=None, limit=20):
    """List recent archived files."""
    from app.services.data.raw_data_archiver import ArchiveCategory
    
    archiver = get_archiver()
    
    cat = None
    if category:
        try:
            cat = ArchiveCategory(category)
        except ValueError:
            print(f"Unknown category: {category}")
            print(f"Valid: {[c.value for c in ArchiveCategory]}")
            return
    
    files = await archiver.list_archives(
        category=cat,
        source=source,
        sport_code=sport,
        limit=limit,
    )
    
    if not files:
        print("No archived files found.")
        return
    
    print(f"\n📁 Recent Archives ({len(files)} files):")
    print(f"{'Name':<50} {'Size':>10} {'Modified':<20}")
    print("-" * 82)
    
    for f in files:
        size_str = f"{f['size_mb']:.2f} MB" if f['size_mb'] >= 1 else f"{f['size_bytes']:,} B"
        print(f"{f['name']:<50} {size_str:>10} {f['modified'][:19]}")


async def cmd_export_db():
    """Export all DB tables to CSV on HDD."""
    from app.core.database import get_session
    from sqlalchemy import text
    
    archiver = get_archiver()
    
    tables_to_export = [
        "games", "teams", "players", "odds", "injuries",
        "game_stats", "player_stats", "predictions",
        "ml_models", "ml_features",
    ]
    
    print("\n📤 Exporting DB tables to HDD archive...")
    
    async with get_session() as session:
        for table_name in tables_to_export:
            try:
                result = await session.execute(
                    text(f"SELECT COUNT(*) FROM {table_name}")
                )
                count = result.scalar()
                
                if count == 0:
                    print(f"  ⏭  {table_name}: empty, skipping")
                    continue
                
                print(f"  📊 {table_name}: {count:,} rows...")
                
                # Fetch in chunks
                chunk_size = 50000
                offset = 0
                all_rows = []
                
                while offset < count:
                    result = await session.execute(
                        text(f"SELECT * FROM {table_name} LIMIT {chunk_size} OFFSET {offset}")
                    )
                    rows = [dict(row._mapping) for row in result.fetchall()]
                    all_rows.extend(rows)
                    offset += chunk_size
                
                paths = await archiver.bulk_export_to_csv(
                    table_name=table_name,
                    rows=all_rows,
                )
                print(f"  ✅ {table_name}: {len(paths)} files created")
                
            except Exception as e:
                print(f"  ❌ {table_name}: {e}")
    
    print("\n✅ Export complete!")


async def cmd_verify():
    """Verify archive integrity (check for corrupted files)."""
    import gzip
    
    archiver = get_archiver()
    base = archiver.base_path
    
    if not base.exists():
        print("Archive path does not exist!")
        return
    
    total = 0
    errors = 0
    
    print("\n🔍 Verifying archive integrity...")
    
    for gz_file in base.rglob("*.json.gz"):
        total += 1
        try:
            with open(gz_file, "rb") as f:
                data = f.read()
            gzip.decompress(data)
        except Exception as e:
            errors += 1
            print(f"  ❌ Corrupted: {gz_file} ({e})")
    
    print(f"\n✅ Verified {total:,} files, {errors} errors")


def main():
    parser = argparse.ArgumentParser(description="ROYALEY Archive Management")
    subparsers = parser.add_subparsers(dest="command", help="Command")
    
    subparsers.add_parser("stats", help="Show storage statistics")
    subparsers.add_parser("report", help="Full storage report")
    
    list_parser = subparsers.add_parser("list", help="List archived files")
    list_parser.add_argument("--category", "-c", help="Filter by category")
    list_parser.add_argument("--source", "-s", help="Filter by source")
    list_parser.add_argument("--sport", help="Filter by sport code")
    list_parser.add_argument("--limit", "-n", type=int, default=20, help="Max files")
    
    subparsers.add_parser("export-db", help="Export DB to HDD CSVs")
    subparsers.add_parser("verify", help="Verify archive integrity")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == "stats":
        asyncio.run(cmd_stats())
    elif args.command == "report":
        asyncio.run(cmd_report())
    elif args.command == "list":
        asyncio.run(cmd_list(
            category=args.category,
            source=args.source,
            sport=args.sport,
            limit=args.limit,
        ))
    elif args.command == "export-db":
        asyncio.run(cmd_export_db())
    elif args.command == "verify":
        asyncio.run(cmd_verify())


if __name__ == "__main__":
    main()
