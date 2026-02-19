"""
ROYALEY - Universal Raw Data Archiver
Saves ALL API responses + bulk data to HDD (16TB /sda-disk) for:
  - Data lineage & audit trail
  - ML training replay
  - Historical analysis
  - Regulatory compliance

Storage Target: 3TB+ across all data types
Storage Location: /sda-disk/raw-data → /app/raw-data (in Docker)

Storage Structure:
/app/raw-data/
├── api-responses/          # Every API call response (gzipped JSON)
│   ├── espn/
│   ├── odds-api/
│   ├── pinnacle/
│   ├── sportsdb/
│   ├── nflfastr/
│   ├── cfbfastr/
│   ├── baseballr/
│   ├── hockeyr/
│   ├── wehoop/
│   ├── hoopr/
│   ├── cfl/
│   ├── action-network/
│   ├── nhl-api/
│   ├── sportsipy/
│   ├── basketball-ref/
│   ├── cfbd/
│   ├── matchstat/
│   ├── realgm/
│   ├── nextgenstats/
│   ├── kaggle/
│   ├── tennis-abstract/
│   ├── polymarket/
│   ├── kalshi/
│   ├── balldontlie/
│   ├── weatherstack/
│   ├── weather/
│   └── tennis/
├── csv/                    # CSV exports (features, datasets, bulk)
│   ├── features/
│   ├── training-data/
│   └── exports/
├── json/                   # Structured JSON dumps (non-API)
│   ├── snapshots/
│   ├── aggregations/
│   └── market-data/
├── binary/                 # Binary files (models, media)
│   ├── models/
│   ├── media/
│   └── downloads/
├── documents/              # Reports, analysis docs
│   ├── reports/
│   └── analysis/
├── datasets/               # Kaggle & bulk datasets
│   ├── kaggle/
│   ├── historical/
│   └── imported/
├── predictions/            # All prediction outputs
│   ├── daily/
│   └── backtests/
├── odds-history/           # Full odds line history
│   ├── opening-lines/
│   ├── line-movements/
│   └── closing-lines/
└── _metadata/              # Archive metadata & indexes
    ├── catalog.json
    └── daily-stats/

File Naming Convention:
  {source}/{sport}/{YYYY}/{MM}/{DD}_{type}_{HHmmss}.{ext}.gz

Supports: JSON, CSV, TSV, binary, MP4, docs, images, any file type
"""

import asyncio
import csv
import gzip
import hashlib
import io
import json
import logging
import os
import shutil
import time
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from functools import wraps

import aiofiles
import aiofiles.os

logger = logging.getLogger(__name__)


class DataFormat(str, Enum):
    """Supported archive data formats."""
    JSON_GZ = "json.gz"
    JSON = "json"
    CSV = "csv"
    CSV_GZ = "csv.gz"
    TSV = "tsv"
    BINARY = "bin"
    MP4 = "mp4"
    PDF = "pdf"
    XLSX = "xlsx"
    PARQUET = "parquet"
    RAW = "raw"


class ArchiveCategory(str, Enum):
    """Top-level archive categories."""
    API_RESPONSES = "api-responses"
    CSV = "csv"
    JSON = "json"
    BINARY = "binary"
    DOCUMENTS = "documents"
    DATASETS = "datasets"
    PREDICTIONS = "predictions"
    ODDS_HISTORY = "odds-history"
    METADATA = "_metadata"

    def __str__(self) -> str:
        """Return value (not 'ArchiveCategory.NAME') for Path compatibility."""
        return self.value

    def __fspath__(self) -> str:
        """Allow direct use in Path operations: Path(...) / ArchiveCategory.X"""
        return self.value


class RawDataArchiver:
    """
    Universal raw data archiver for ALL Royaley data sources.
    
    Saves every API response and data artifact to the 16TB HDD.
    Goal: Accumulate 3TB+ of raw data for ML training and analysis.
    
    Features:
    - Auto-archive every API response (integrated into BaseCollector)
    - Multi-format support (JSON, CSV, binary, media, docs)
    - Gzip compression for text data (5-10x savings)
    - SHA256 checksums for data integrity
    - Daily/monthly storage statistics
    - Catalog index for fast lookup
    - Async I/O for non-blocking writes
    
    Usage:
        archiver = get_archiver()
        
        # Archive API response (called automatically by BaseCollector)
        await archiver.archive_api_response("espn", "NFL", data, "scoreboard")
        
        # Archive CSV data
        await archiver.archive_csv("features", "NFL", rows, headers)
        
        # Archive binary file
        await archiver.archive_binary("models", "nfl_model_v2.pkl", binary_data)
        
        # Get storage stats
        stats = archiver.get_storage_stats()
    """
    
    # All 27 collector source names
    COLLECTOR_SOURCES = [
        "espn", "odds-api", "pinnacle", "tennis", "weather",
        "sportsdb", "nflfastr", "cfbfastr", "baseballr", "hockeyr",
        "wehoop", "hoopr", "cfl", "action-network", "nhl-api",
        "sportsipy", "basketball-ref", "cfbd", "matchstat", "realgm",
        "nextgenstats", "kaggle", "tennis-abstract", "polymarket",
        "kalshi", "balldontlie", "weatherstack",
    ]
    
    def __init__(self, base_path: str = "/app/raw-data"):
        self.base_path = Path(base_path)
        self.enabled = True
        self._stats_cache: Optional[Dict] = None
        self._stats_cache_time: float = 0
        self._daily_bytes_written: int = 0
        self._daily_files_written: int = 0
        self._total_archives_session: int = 0
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create complete archive directory structure on HDD."""
        try:
            # API response directories for all 27 collectors
            for source in self.COLLECTOR_SOURCES:
                (self.base_path / ArchiveCategory.API_RESPONSES / source).mkdir(
                    parents=True, exist_ok=True
                )
            
            # CSV directories
            for sub in ["features", "training-data", "exports"]:
                (self.base_path / ArchiveCategory.CSV / sub).mkdir(
                    parents=True, exist_ok=True
                )
            
            # JSON directories
            for sub in ["snapshots", "aggregations", "market-data"]:
                (self.base_path / ArchiveCategory.JSON / sub).mkdir(
                    parents=True, exist_ok=True
                )
            
            # Binary directories
            for sub in ["models", "media", "downloads"]:
                (self.base_path / ArchiveCategory.BINARY / sub).mkdir(
                    parents=True, exist_ok=True
                )
            
            # Document directories
            for sub in ["reports", "analysis"]:
                (self.base_path / ArchiveCategory.DOCUMENTS / sub).mkdir(
                    parents=True, exist_ok=True
                )
            
            # Dataset directories
            for sub in ["kaggle", "historical", "imported"]:
                (self.base_path / ArchiveCategory.DATASETS / sub).mkdir(
                    parents=True, exist_ok=True
                )
            
            # Prediction directories
            for sub in ["daily", "backtests"]:
                (self.base_path / ArchiveCategory.PREDICTIONS / sub).mkdir(
                    parents=True, exist_ok=True
                )
            
            # Odds history directories
            for sub in ["opening-lines", "line-movements", "closing-lines"]:
                (self.base_path / ArchiveCategory.ODDS_HISTORY / sub).mkdir(
                    parents=True, exist_ok=True
                )
            
            # Metadata
            (self.base_path / ArchiveCategory.METADATA / "daily-stats").mkdir(
                parents=True, exist_ok=True
            )
            
            logger.info(f"📁 Raw data archive initialized at {self.base_path}")
            
        except PermissionError as e:
            logger.error(f"❌ Cannot create archive directories (permission denied): {e}")
            self.enabled = False
        except Exception as e:
            logger.error(f"❌ Failed to initialize archive directories: {e}")
            self.enabled = False
    
    # =========================================================================
    # CORE: API Response Archiving (auto-called by BaseCollector)
    # =========================================================================
    
    async def archive_api_response(
        self,
        source: str,
        sport_code: Optional[str],
        data: Any,
        data_type: str = "response",
        endpoint: str = "",
        params: Optional[Dict] = None,
        response_status: int = 200,
        response_headers: Optional[Dict] = None,
    ) -> Optional[Path]:
        """
        Archive a raw API response to HDD. This is the PRIMARY archive method,
        called automatically by BaseCollector._make_request() for every API call.
        
        Path: /app/raw-data/api-responses/{source}/{sport}/{YYYY}/{MM}/{DD}_{type}_{HHmmss}.json.gz
        
        Args:
            source: Collector name (espn, odds-api, pinnacle, etc.)
            sport_code: Sport code (NFL, NBA, etc.) or None
            data: Raw API response data (dict, list, or string)
            data_type: Type of data (scoreboard, teams, standings, odds, etc.)
            endpoint: API endpoint that was called
            params: Query parameters used
            response_status: HTTP response status code
            response_headers: Response headers (for rate limit tracking, etc.)
            
        Returns:
            Path to archived file, or None on failure
        """
        if not self.enabled:
            return None
        
        try:
            now = datetime.utcnow()
            
            # Normalize source name
            source_clean = source.lower().replace("_", "-")
            sport_clean = (sport_code or "general").upper()
            
            # Build directory path
            dir_path = (
                self.base_path / ArchiveCategory.API_RESPONSES / source_clean /
                sport_clean / now.strftime("%Y") / now.strftime("%m")
            )
            dir_path.mkdir(parents=True, exist_ok=True)
            
            # Build filename: DD_type_HHmmss.json.gz
            timestamp = now.strftime("%d_%H%M%S")
            safe_type = data_type.replace("/", "_").replace(" ", "_")[:50]
            file_name = f"{timestamp}_{safe_type}.json.gz"
            file_path = dir_path / file_name
            
            # Handle filename collision (same second)
            counter = 1
            while file_path.exists():
                file_name = f"{timestamp}_{safe_type}_{counter}.json.gz"
                file_path = dir_path / file_name
                counter += 1
            
            # Build archive envelope with full metadata
            archived_data = {
                "_archive_meta": {
                    "source": source_clean,
                    "sport": sport_clean,
                    "data_type": data_type,
                    "endpoint": endpoint,
                    "params": params,
                    "response_status": response_status,
                    "archived_at": now.isoformat(),
                    "archive_version": "2.0",
                    "file_path": str(file_path),
                },
                "data": data,
            }
            
            # Write gzipped JSON
            bytes_written = await self._write_gzipped_json(file_path, archived_data)
            
            # Update stats
            self._daily_bytes_written += bytes_written
            self._daily_files_written += 1
            self._total_archives_session += 1
            
            if self._total_archives_session % 100 == 0:
                logger.info(
                    f"💾 Archive milestone: {self._total_archives_session} files, "
                    f"{self._daily_bytes_written / (1024**2):.1f} MB today"
                )
            
            logger.debug(f"💾 Archived {source_clean}/{sport_clean}/{data_type} → {file_path} ({bytes_written} bytes)")
            return file_path
            
        except Exception as e:
            logger.error(f"❌ Failed to archive API response [{source}/{data_type}]: {e}")
            return None
    
    # =========================================================================
    # CSV Archiving
    # =========================================================================
    
    async def archive_csv(
        self,
        subcategory: str,
        name: str,
        rows: List[Dict[str, Any]],
        headers: Optional[List[str]] = None,
        sport_code: Optional[str] = None,
        compress: bool = True,
    ) -> Optional[Path]:
        """
        Archive data as CSV to HDD.
        
        Args:
            subcategory: Sub-folder (features, training-data, exports)
            name: Descriptive name for the file
            rows: List of dicts (each dict = one row)
            headers: Column headers (auto-detected from first row if None)
            sport_code: Optional sport code for organization
            compress: Whether to gzip the CSV
            
        Returns:
            Path to archived file
        """
        if not self.enabled or not rows:
            return None
        
        try:
            now = datetime.utcnow()
            
            # Build path
            dir_path = self.base_path / ArchiveCategory.CSV / subcategory
            if sport_code:
                dir_path = dir_path / sport_code.upper()
            dir_path = dir_path / now.strftime("%Y") / now.strftime("%m")
            dir_path.mkdir(parents=True, exist_ok=True)
            
            # Determine headers
            if headers is None:
                headers = list(rows[0].keys())
            
            # Build CSV in memory
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=headers, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
            csv_content = output.getvalue()
            
            # Write file
            safe_name = name.replace("/", "_").replace(" ", "_")
            timestamp = now.strftime("%d_%H%M%S")
            
            if compress:
                file_name = f"{timestamp}_{safe_name}.csv.gz"
                file_path = dir_path / file_name
                compressed = gzip.compress(csv_content.encode("utf-8"))
                async with aiofiles.open(file_path, "wb") as f:
                    await f.write(compressed)
                bytes_written = len(compressed)
            else:
                file_name = f"{timestamp}_{safe_name}.csv"
                file_path = dir_path / file_name
                async with aiofiles.open(file_path, "w") as f:
                    await f.write(csv_content)
                bytes_written = len(csv_content.encode("utf-8"))
            
            self._daily_bytes_written += bytes_written
            self._daily_files_written += 1
            
            logger.info(f"📊 Archived CSV: {file_path} ({len(rows)} rows, {bytes_written} bytes)")
            return file_path
            
        except Exception as e:
            logger.error(f"❌ Failed to archive CSV [{name}]: {e}")
            return None
    
    # =========================================================================
    # JSON Data Archiving (non-API, structured dumps)
    # =========================================================================
    
    async def archive_json(
        self,
        subcategory: str,
        name: str,
        data: Any,
        sport_code: Optional[str] = None,
        compress: bool = True,
    ) -> Optional[Path]:
        """Archive structured JSON data (snapshots, aggregations, market data)."""
        if not self.enabled:
            return None
        
        try:
            now = datetime.utcnow()
            
            dir_path = self.base_path / ArchiveCategory.JSON / subcategory
            if sport_code:
                dir_path = dir_path / sport_code.upper()
            dir_path = dir_path / now.strftime("%Y") / now.strftime("%m")
            dir_path.mkdir(parents=True, exist_ok=True)
            
            safe_name = name.replace("/", "_").replace(" ", "_")
            timestamp = now.strftime("%d_%H%M%S")
            
            if compress:
                file_name = f"{timestamp}_{safe_name}.json.gz"
                file_path = dir_path / file_name
                bytes_written = await self._write_gzipped_json(file_path, data)
            else:
                file_name = f"{timestamp}_{safe_name}.json"
                file_path = dir_path / file_name
                json_str = json.dumps(data, indent=2, default=str)
                async with aiofiles.open(file_path, "w") as f:
                    await f.write(json_str)
                bytes_written = len(json_str.encode("utf-8"))
            
            self._daily_bytes_written += bytes_written
            self._daily_files_written += 1
            
            logger.debug(f"📄 Archived JSON: {file_path} ({bytes_written} bytes)")
            return file_path
            
        except Exception as e:
            logger.error(f"❌ Failed to archive JSON [{name}]: {e}")
            return None
    
    # =========================================================================
    # Binary / Media Archiving
    # =========================================================================
    
    async def archive_binary(
        self,
        subcategory: str,
        filename: str,
        data: bytes,
        sport_code: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> Optional[Path]:
        """
        Archive binary data (models, media, MP4, images, etc.) to HDD.
        
        Args:
            subcategory: Sub-folder (models, media, downloads)
            filename: Original filename with extension
            data: Raw binary data
            sport_code: Optional sport code
            metadata: Optional metadata dict (saved alongside as .meta.json)
        """
        if not self.enabled:
            return None
        
        try:
            now = datetime.utcnow()
            
            dir_path = self.base_path / ArchiveCategory.BINARY / subcategory
            if sport_code:
                dir_path = dir_path / sport_code.upper()
            dir_path = dir_path / now.strftime("%Y") / now.strftime("%m")
            dir_path.mkdir(parents=True, exist_ok=True)
            
            # Preserve original extension
            timestamp = now.strftime("%d_%H%M%S")
            safe_name = filename.replace("/", "_").replace(" ", "_")
            file_path = dir_path / f"{timestamp}_{safe_name}"
            
            # Write binary data
            async with aiofiles.open(file_path, "wb") as f:
                await f.write(data)
            
            bytes_written = len(data)
            
            # Write metadata sidecar if provided
            if metadata:
                meta_path = file_path.with_suffix(file_path.suffix + ".meta.json")
                meta = {
                    "original_filename": filename,
                    "size_bytes": bytes_written,
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "archived_at": now.isoformat(),
                    "sport": sport_code,
                    **metadata,
                }
                async with aiofiles.open(meta_path, "w") as f:
                    await f.write(json.dumps(meta, indent=2, default=str))
            
            self._daily_bytes_written += bytes_written
            self._daily_files_written += 1
            
            logger.info(f"📦 Archived binary: {file_path} ({bytes_written / (1024**2):.1f} MB)")
            return file_path
            
        except Exception as e:
            logger.error(f"❌ Failed to archive binary [{filename}]: {e}")
            return None
    
    # =========================================================================
    # File Copy Archiving (for external files: docs, datasets, etc.)
    # =========================================================================
    
    async def archive_file(
        self,
        source_path: str,
        category: ArchiveCategory = ArchiveCategory.DOCUMENTS,
        subcategory: str = "imported",
        sport_code: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> Optional[Path]:
        """
        Archive an existing file by copying it to the HDD archive.
        Supports any file type: PDF, XLSX, DOCX, MP4, images, etc.
        """
        if not self.enabled:
            return None
        
        try:
            source = Path(source_path)
            if not source.exists():
                logger.error(f"Source file not found: {source_path}")
                return None
            
            now = datetime.utcnow()
            
            dir_path = self.base_path / category / subcategory
            if sport_code:
                dir_path = dir_path / sport_code.upper()
            dir_path = dir_path / now.strftime("%Y") / now.strftime("%m")
            dir_path.mkdir(parents=True, exist_ok=True)
            
            timestamp = now.strftime("%d_%H%M%S")
            dest_name = f"{timestamp}_{source.name}"
            dest_path = dir_path / dest_name
            
            # Async copy
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, shutil.copy2, str(source), str(dest_path))
            
            file_size = dest_path.stat().st_size
            self._daily_bytes_written += file_size
            self._daily_files_written += 1
            
            # Write metadata sidecar
            if metadata:
                meta_path = dest_path.with_suffix(dest_path.suffix + ".meta.json")
                meta = {
                    "original_path": str(source_path),
                    "original_name": source.name,
                    "size_bytes": file_size,
                    "archived_at": now.isoformat(),
                    **metadata,
                }
                async with aiofiles.open(meta_path, "w") as f:
                    await f.write(json.dumps(meta, indent=2, default=str))
            
            logger.info(f"📎 Archived file: {dest_path} ({file_size / (1024**2):.1f} MB)")
            return dest_path
            
        except Exception as e:
            logger.error(f"❌ Failed to archive file [{source_path}]: {e}")
            return None
    
    # =========================================================================
    # Odds History Archiving
    # =========================================================================
    
    async def archive_odds_snapshot(
        self,
        sport_code: str,
        odds_data: List[Dict[str, Any]],
        snapshot_type: str = "line-movements",
    ) -> Optional[Path]:
        """Archive odds/line data for historical analysis."""
        if not self.enabled or not odds_data:
            return None
        
        try:
            now = datetime.utcnow()
            
            dir_path = (
                self.base_path / ArchiveCategory.ODDS_HISTORY / snapshot_type /
                sport_code.upper() / now.strftime("%Y") / now.strftime("%m")
            )
            dir_path.mkdir(parents=True, exist_ok=True)
            
            timestamp = now.strftime("%d_%H%M%S")
            file_path = dir_path / f"{timestamp}_odds.json.gz"
            
            archive_data = {
                "_meta": {
                    "sport": sport_code,
                    "type": snapshot_type,
                    "count": len(odds_data),
                    "archived_at": now.isoformat(),
                },
                "odds": odds_data,
            }
            
            bytes_written = await self._write_gzipped_json(file_path, archive_data)
            self._daily_bytes_written += bytes_written
            self._daily_files_written += 1
            
            logger.debug(f"📈 Archived odds: {file_path} ({len(odds_data)} records)")
            return file_path
            
        except Exception as e:
            logger.error(f"❌ Failed to archive odds [{sport_code}]: {e}")
            return None
    
    # =========================================================================
    # Prediction Archiving
    # =========================================================================
    
    async def archive_predictions(
        self,
        sport_code: str,
        predictions: List[Dict[str, Any]],
        prediction_type: str = "daily",
        model_info: Optional[Dict] = None,
    ) -> Optional[Path]:
        """Archive prediction outputs for tracking and analysis."""
        if not self.enabled or not predictions:
            return None
        
        try:
            now = datetime.utcnow()
            
            dir_path = (
                self.base_path / ArchiveCategory.PREDICTIONS / prediction_type /
                sport_code.upper() / now.strftime("%Y") / now.strftime("%m")
            )
            dir_path.mkdir(parents=True, exist_ok=True)
            
            timestamp = now.strftime("%d_%H%M%S")
            file_path = dir_path / f"{timestamp}_predictions.json.gz"
            
            archive_data = {
                "_meta": {
                    "sport": sport_code,
                    "type": prediction_type,
                    "count": len(predictions),
                    "model_info": model_info,
                    "archived_at": now.isoformat(),
                },
                "predictions": predictions,
            }
            
            bytes_written = await self._write_gzipped_json(file_path, archive_data)
            self._daily_bytes_written += bytes_written
            self._daily_files_written += 1
            
            return file_path
            
        except Exception as e:
            logger.error(f"❌ Failed to archive predictions: {e}")
            return None
    
    # =========================================================================
    # Dataset Archiving (Kaggle, bulk imports)
    # =========================================================================
    
    async def archive_dataset(
        self,
        source: str,
        name: str,
        data: Union[bytes, str, Dict, List],
        file_extension: str = "csv",
        sport_code: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> Optional[Path]:
        """
        Archive a downloaded dataset (Kaggle, bulk CSV, etc.)
        """
        if not self.enabled:
            return None
        
        try:
            now = datetime.utcnow()
            
            dir_path = self.base_path / ArchiveCategory.DATASETS / source
            if sport_code:
                dir_path = dir_path / sport_code.upper()
            dir_path = dir_path / now.strftime("%Y") / now.strftime("%m")
            dir_path.mkdir(parents=True, exist_ok=True)
            
            safe_name = name.replace("/", "_").replace(" ", "_")
            timestamp = now.strftime("%d_%H%M%S")
            file_path = dir_path / f"{timestamp}_{safe_name}.{file_extension}"
            
            if isinstance(data, bytes):
                async with aiofiles.open(file_path, "wb") as f:
                    await f.write(data)
                bytes_written = len(data)
            elif isinstance(data, str):
                async with aiofiles.open(file_path, "w") as f:
                    await f.write(data)
                bytes_written = len(data.encode("utf-8"))
            else:
                # Dict or List → JSON
                json_str = json.dumps(data, indent=2, default=str)
                async with aiofiles.open(file_path, "w") as f:
                    await f.write(json_str)
                bytes_written = len(json_str.encode("utf-8"))
            
            # Metadata sidecar
            if metadata:
                meta_path = file_path.with_suffix(file_path.suffix + ".meta.json")
                meta = {
                    "dataset_name": name,
                    "source": source,
                    "size_bytes": bytes_written,
                    "sport": sport_code,
                    "archived_at": now.isoformat(),
                    **metadata,
                }
                async with aiofiles.open(meta_path, "w") as f:
                    await f.write(json.dumps(meta, indent=2, default=str))
            
            self._daily_bytes_written += bytes_written
            self._daily_files_written += 1
            
            logger.info(
                f"📥 Archived dataset: {file_path} "
                f"({bytes_written / (1024**2):.1f} MB)"
            )
            return file_path
            
        except Exception as e:
            logger.error(f"❌ Failed to archive dataset [{name}]: {e}")
            return None
    
    # =========================================================================
    # Bulk Export: DB → HDD
    # =========================================================================
    
    async def bulk_export_to_csv(
        self,
        table_name: str,
        rows: List[Dict[str, Any]],
        sport_code: Optional[str] = None,
        chunk_size: int = 50000,
    ) -> List[Path]:
        """
        Bulk export database records to CSV files on HDD.
        Splits large datasets into chunks for manageable file sizes.
        """
        if not self.enabled or not rows:
            return []
        
        paths = []
        headers = list(rows[0].keys())
        
        for i in range(0, len(rows), chunk_size):
            chunk = rows[i:i + chunk_size]
            chunk_num = i // chunk_size + 1
            name = f"{table_name}_chunk{chunk_num:04d}"
            
            path = await self.archive_csv(
                subcategory="exports",
                name=name,
                rows=chunk,
                headers=headers,
                sport_code=sport_code,
                compress=True,
            )
            if path:
                paths.append(path)
        
        logger.info(
            f"📊 Bulk export: {table_name} → {len(paths)} files, "
            f"{len(rows)} total rows"
        )
        return paths
    
    # =========================================================================
    # Storage Statistics & Monitoring
    # =========================================================================
    
    def get_storage_stats(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Get comprehensive storage statistics.
        Cached for 5 minutes to avoid expensive disk scans.
        """
        cache_ttl = 300
        if (
            not force_refresh
            and self._stats_cache
            and (time.time() - self._stats_cache_time) < cache_ttl
        ):
            return self._stats_cache
        
        stats = {
            "archive_path": str(self.base_path),
            "enabled": self.enabled,
            "total_size_bytes": 0,
            "total_size_gb": 0.0,
            "total_size_tb": 0.0,
            "total_files": 0,
            "target_size_tb": 3.0,
            "progress_pct": 0.0,
            "session_stats": {
                "bytes_written": self._daily_bytes_written,
                "files_written": self._daily_files_written,
                "total_archives": self._total_archives_session,
            },
            "categories": {},
            "top_sources": {},
            "disk_info": self._get_disk_info(),
            "generated_at": datetime.utcnow().isoformat(),
        }
        
        if not self.base_path.exists():
            self._stats_cache = stats
            self._stats_cache_time = time.time()
            return stats
        
        # Scan each category
        for category in ArchiveCategory:
            cat_path = self.base_path / category
            if cat_path.exists():
                cat_stats = self._scan_directory(cat_path)
                stats["categories"][category.value] = cat_stats
                stats["total_size_bytes"] += cat_stats["size_bytes"]
                stats["total_files"] += cat_stats["file_count"]
        
        # Scan API response sources for detailed breakdown
        api_path = self.base_path / ArchiveCategory.API_RESPONSES
        if api_path.exists():
            for source_dir in sorted(api_path.iterdir()):
                if source_dir.is_dir() and source_dir.name != "lost+found":
                    source_stats = self._scan_directory(source_dir)
                    stats["top_sources"][source_dir.name] = {
                        "files": source_stats["file_count"],
                        "size_mb": source_stats["size_bytes"] / (1024**2),
                    }
        
        stats["total_size_gb"] = stats["total_size_bytes"] / (1024**3)
        stats["total_size_tb"] = stats["total_size_bytes"] / (1024**4)
        stats["progress_pct"] = min(
            100.0, (stats["total_size_tb"] / stats["target_size_tb"]) * 100
        )
        
        self._stats_cache = stats
        self._stats_cache_time = time.time()
        
        return stats
    
    def _scan_directory(self, directory: Path) -> Dict[str, Any]:
        """Recursively scan a directory for size and file count."""
        total_size = 0
        file_count = 0
        
        try:
            for item in directory.rglob("*"):
                if item.is_file():
                    try:
                        total_size += item.stat().st_size
                        file_count += 1
                    except OSError:
                        pass
        except PermissionError:
            pass
        
        return {
            "size_bytes": total_size,
            "size_gb": total_size / (1024**3),
            "file_count": file_count,
        }
    
    def _get_disk_info(self) -> Dict[str, Any]:
        """Get HDD disk usage info."""
        try:
            usage = shutil.disk_usage(str(self.base_path))
            return {
                "total_tb": usage.total / (1024**4),
                "used_tb": usage.used / (1024**4),
                "free_tb": usage.free / (1024**4),
                "used_pct": (usage.used / usage.total) * 100,
            }
        except Exception:
            return {"error": "Could not read disk info"}
    
    def print_storage_report(self) -> str:
        """Generate a human-readable storage report."""
        stats = self.get_storage_stats(force_refresh=True)
        
        lines = [
            "",
            "=" * 70,
            "  ROYALEY RAW DATA ARCHIVE - STORAGE REPORT",
            "=" * 70,
            f"  Archive Path:  {stats['archive_path']}",
            f"  Status:        {'✅ ENABLED' if stats['enabled'] else '❌ DISABLED'}",
            f"  Total Size:    {stats['total_size_gb']:.2f} GB ({stats['total_size_tb']:.3f} TB)",
            f"  Total Files:   {stats['total_files']:,}",
            f"  Target:        {stats['target_size_tb']:.1f} TB",
            f"  Progress:      {stats['progress_pct']:.1f}%",
            "",
            "  📁 Categories:",
        ]
        
        for cat_name, cat_stats in sorted(stats.get("categories", {}).items()):
            lines.append(
                f"    {cat_name:<20} {cat_stats['size_gb']:>8.2f} GB  "
                f"({cat_stats['file_count']:>7,} files)"
            )
        
        if stats.get("top_sources"):
            lines.append("")
            lines.append("  📡 API Sources:")
            for source, info in sorted(
                stats["top_sources"].items(),
                key=lambda x: x[1]["size_mb"],
                reverse=True,
            ):
                lines.append(
                    f"    {source:<22} {info['size_mb']:>8.1f} MB  "
                    f"({info['files']:>6,} files)"
                )
        
        disk = stats.get("disk_info", {})
        if "total_tb" in disk:
            lines.extend([
                "",
                "  💿 HDD Disk:",
                f"    Total:  {disk['total_tb']:.1f} TB",
                f"    Used:   {disk['used_tb']:.1f} TB ({disk['used_pct']:.1f}%)",
                f"    Free:   {disk['free_tb']:.1f} TB",
            ])
        
        session = stats.get("session_stats", {})
        lines.extend([
            "",
            "  📊 This Session:",
            f"    Files Written:   {session.get('files_written', 0):,}",
            f"    Bytes Written:   {session.get('bytes_written', 0) / (1024**2):.1f} MB",
            f"    Total Archives:  {session.get('total_archives', 0):,}",
            "=" * 70,
            "",
        ])
        
        report = "\n".join(lines)
        logger.info(report)
        return report
    
    # =========================================================================
    # Daily Statistics Persistence
    # =========================================================================
    
    async def save_daily_stats(self):
        """Save daily archival statistics to disk."""
        try:
            now = datetime.utcnow()
            stats_dir = self.base_path / ArchiveCategory.METADATA / "daily-stats"
            stats_dir.mkdir(parents=True, exist_ok=True)
            
            stats_file = stats_dir / f"{now.strftime('%Y-%m-%d')}.json"
            
            daily_stats = {
                "date": now.strftime("%Y-%m-%d"),
                "bytes_written": self._daily_bytes_written,
                "files_written": self._daily_files_written,
                "total_archives_session": self._total_archives_session,
                "storage_stats": self.get_storage_stats(force_refresh=True),
            }
            
            async with aiofiles.open(stats_file, "w") as f:
                await f.write(json.dumps(daily_stats, indent=2, default=str))
            
            logger.info(f"📊 Saved daily stats: {stats_file}")
            
        except Exception as e:
            logger.error(f"Failed to save daily stats: {e}")
    
    # =========================================================================
    # Read / Replay
    # =========================================================================
    
    async def read_archived_data(self, path: Union[str, Path]) -> Optional[Any]:
        """Read archived data from file (handles gzipped and plain)."""
        try:
            path = Path(path)
            
            if path.suffix == ".gz":
                async with aiofiles.open(path, "rb") as f:
                    compressed = await f.read()
                json_str = gzip.decompress(compressed).decode("utf-8")
                return json.loads(json_str)
            elif path.suffix == ".json":
                async with aiofiles.open(path, "r") as f:
                    content = await f.read()
                return json.loads(content)
            elif path.suffix == ".csv":
                async with aiofiles.open(path, "r") as f:
                    content = await f.read()
                reader = csv.DictReader(io.StringIO(content))
                return list(reader)
            else:
                async with aiofiles.open(path, "rb") as f:
                    return await f.read()
                    
        except Exception as e:
            logger.error(f"Failed to read archived data [{path}]: {e}")
            return None
    
    async def list_archives(
        self,
        category: Optional[ArchiveCategory] = None,
        source: Optional[str] = None,
        sport_code: Optional[str] = None,
        date: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """List archived files matching criteria."""
        search_path = self.base_path
        
        if category:
            search_path = search_path / category
        if source:
            search_path = search_path / source.lower().replace("_", "-")
        if sport_code:
            search_path = search_path / sport_code.upper()
        if date:
            search_path = search_path / date.strftime("%Y") / date.strftime("%m")
        
        results = []
        try:
            if search_path.exists():
                for f in sorted(search_path.rglob("*"), reverse=True):
                    if f.is_file() and not f.name.endswith(".meta.json"):
                        try:
                            stat = f.stat()
                            results.append({
                                "path": str(f),
                                "name": f.name,
                                "size_bytes": stat.st_size,
                                "size_mb": stat.st_size / (1024**2),
                                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                            })
                        except OSError:
                            pass
                    
                    if len(results) >= limit:
                        break
        except Exception as e:
            logger.error(f"Failed to list archives: {e}")
        
        return results
    
    # =========================================================================
    # Internal Helpers
    # =========================================================================
    
    async def _write_gzipped_json(self, path: Path, data: Any) -> int:
        """Write compressed JSON to file. Returns bytes written."""
        json_str = json.dumps(data, indent=None, default=str, ensure_ascii=False)
        compressed = gzip.compress(json_str.encode("utf-8"), compresslevel=6)
        
        async with aiofiles.open(path, "wb") as f:
            await f.write(compressed)
        
        return len(compressed)


# =============================================================================
# Global Archiver Instance
# =============================================================================

_archiver: Optional[RawDataArchiver] = None


def get_archiver() -> RawDataArchiver:
    """Get global archiver instance (singleton)."""
    global _archiver
    if _archiver is None:
        _archiver = RawDataArchiver()
    return _archiver


def archive_response(source: str):
    """
    Decorator to automatically archive API responses.
    
    Usage:
        @archive_response("espn")
        async def fetch_games(sport_code: str, date: datetime):
            ...
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)
            
            try:
                archiver = get_archiver()
                sport_code = kwargs.get("sport_code") or (args[0] if args else None)
                data_type = kwargs.get("data_type", func.__name__)
                
                await archiver.archive_api_response(
                    source=source,
                    sport_code=sport_code if isinstance(sport_code, str) else None,
                    data=result,
                    data_type=data_type,
                )
            except Exception as e:
                logger.error(f"Archive decorator error [{source}]: {e}")
            
            return result
        return wrapper
    return decorator