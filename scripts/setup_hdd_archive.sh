#!/bin/bash
# ============================================================================
# ROYALEY - HDD Raw Data Archive Setup Script
# Initializes the 16TB HDD archive directory structure
# Run once on the server: bash scripts/setup_hdd_archive.sh
# ============================================================================

set -e

HDD_PATH="/sda-disk"
ARCHIVE_PATH="${HDD_PATH}/raw-data"

echo "=============================================="
echo "  ROYALEY - HDD Archive Setup"
echo "=============================================="
echo ""

# Check HDD is mounted
if ! mountpoint -q "${HDD_PATH}" 2>/dev/null; then
    if [ ! -d "${HDD_PATH}" ]; then
        echo "❌ ERROR: ${HDD_PATH} is not mounted!"
        echo "   Mount the 16TB HDD first."
        exit 1
    fi
fi

echo "💿 HDD Status:"
df -h "${HDD_PATH}"
echo ""

# Create complete directory structure
echo "📁 Creating archive directory structure..."

# API response directories for all 27 collectors
COLLECTORS=(
    "espn" "odds-api" "pinnacle" "tennis" "weather"
    "sportsdb" "nflfastr" "cfbfastr" "baseballr" "hockeyr"
    "wehoop" "hoopr" "cfl" "action-network" "nhl-api"
    "sportsipy" "basketball-ref" "cfbd" "matchstat" "realgm"
    "nextgenstats" "kaggle" "tennis-abstract" "polymarket"
    "kalshi" "balldontlie" "weatherstack"
)

for collector in "${COLLECTORS[@]}"; do
    mkdir -p "${ARCHIVE_PATH}/api-responses/${collector}"
done

# CSV directories
mkdir -p "${ARCHIVE_PATH}/csv/features"
mkdir -p "${ARCHIVE_PATH}/csv/training-data"
mkdir -p "${ARCHIVE_PATH}/csv/exports"

# JSON directories
mkdir -p "${ARCHIVE_PATH}/json/snapshots"
mkdir -p "${ARCHIVE_PATH}/json/aggregations"
mkdir -p "${ARCHIVE_PATH}/json/market-data"

# Binary directories
mkdir -p "${ARCHIVE_PATH}/binary/models"
mkdir -p "${ARCHIVE_PATH}/binary/media"
mkdir -p "${ARCHIVE_PATH}/binary/downloads"

# Document directories
mkdir -p "${ARCHIVE_PATH}/documents/reports"
mkdir -p "${ARCHIVE_PATH}/documents/analysis"

# Dataset directories
mkdir -p "${ARCHIVE_PATH}/datasets/kaggle"
mkdir -p "${ARCHIVE_PATH}/datasets/historical"
mkdir -p "${ARCHIVE_PATH}/datasets/imported"

# Prediction directories
mkdir -p "${ARCHIVE_PATH}/predictions/daily"
mkdir -p "${ARCHIVE_PATH}/predictions/backtests"

# Odds history directories
mkdir -p "${ARCHIVE_PATH}/odds-history/opening-lines"
mkdir -p "${ARCHIVE_PATH}/odds-history/line-movements"
mkdir -p "${ARCHIVE_PATH}/odds-history/closing-lines"

# Metadata directories
mkdir -p "${ARCHIVE_PATH}/_metadata/daily-stats"

# Set permissions (owned by Docker user)
echo "🔐 Setting permissions..."
chmod -R 777 "${ARCHIVE_PATH}"

echo ""
echo "✅ Archive directory structure created!"
echo ""

# Show tree
echo "📂 Archive Structure:"
find "${ARCHIVE_PATH}" -maxdepth 2 -type d | head -50 | sed 's/^/  /'
echo "  ..."
echo ""

# Count directories
DIR_COUNT=$(find "${ARCHIVE_PATH}" -type d | wc -l)
echo "📊 Summary:"
echo "  Total directories: ${DIR_COUNT}"
echo "  Archive path:      ${ARCHIVE_PATH}"
echo "  Docker mount:      /app/raw-data"
echo ""

echo "🚀 Ready! Start collecting data with:"
echo "   docker compose exec api python -c \"from app.services.data import get_archiver; print(get_archiver().print_storage_report())\""
echo ""
echo "   Monitor with:"
echo "   docker compose exec api python -m app.cli.archive stats"
echo "   docker compose exec api python -m app.cli.archive report"
echo ""
echo "=============================================="
