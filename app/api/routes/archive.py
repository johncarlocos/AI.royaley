"""
ROYALEY - Archive Storage API Routes
Monitor HDD raw data archive via REST API.

Endpoints:
    GET /api/v1/archive/stats     - Storage statistics
    GET /api/v1/archive/report    - Full storage report
    GET /api/v1/archive/files     - List archived files
"""

from fastapi import APIRouter, Query
from typing import Optional, Dict, Any, List

from app.services.data.raw_data_archiver import get_archiver, ArchiveCategory

router = APIRouter(prefix="/archive", tags=["archive"])


@router.get("/stats", response_model=Dict[str, Any])
async def get_archive_stats(refresh: bool = Query(False, description="Force refresh")):
    """
    Get archive storage statistics.
    Shows total size, per-category breakdown, progress toward 3TB target.
    """
    archiver = get_archiver()
    return archiver.get_storage_stats(force_refresh=refresh)


@router.get("/report")
async def get_archive_report():
    """Get human-readable storage report."""
    archiver = get_archiver()
    report = archiver.print_storage_report()
    return {"report": report}


@router.get("/files", response_model=List[Dict[str, Any]])
async def list_archive_files(
    category: Optional[str] = Query(None, description="Category filter"),
    source: Optional[str] = Query(None, description="Source filter (e.g. espn, odds-api)"),
    sport: Optional[str] = Query(None, description="Sport code filter (e.g. NFL, NBA)"),
    limit: int = Query(50, ge=1, le=500, description="Max files to return"),
):
    """List archived files with optional filters."""
    archiver = get_archiver()
    
    cat = None
    if category:
        try:
            cat = ArchiveCategory(category)
        except ValueError:
            return []
    
    return await archiver.list_archives(
        category=cat,
        source=source,
        sport_code=sport,
        limit=limit,
    )


@router.get("/disk")
async def get_disk_info():
    """Get HDD disk usage information."""
    archiver = get_archiver()
    stats = archiver.get_storage_stats()
    return {
        "disk": stats.get("disk_info", {}),
        "archive_size_gb": stats.get("total_size_gb", 0),
        "archive_size_tb": stats.get("total_size_tb", 0),
        "target_tb": stats.get("target_size_tb", 3.0),
        "progress_pct": stats.get("progress_pct", 0),
    }
