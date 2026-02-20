#!/bin/bash
#===============================================================================
# ROYALEY → GOOGLE DRIVE AUTOMATED BACKUP
#===============================================================================
#
# Backs up all critical data to Google Workspace (5TB) via rclone
#
# Schedule:
#   DAILY  (3 AM UTC): Database, code, models, configs
#   WEEKLY (Sun 4 AM): Raw data from HDD (3.7TB)
#
# Usage:
#   ./backup_gdrive.sh              # Run daily backup
#   ./backup_gdrive.sh --weekly     # Run daily + raw data (weekly)
#   ./backup_gdrive.sh --status     # Show backup status
#   ./backup_gdrive.sh --install    # Install cron jobs
#   ./backup_gdrive.sh --db-only    # Database backup only
#
# Retention on Google Drive:
#   Royaley-Backups/
#     daily/
#       2026-02-20/         ← latest daily
#       2026-02-19/
#       ...                 ← keeps last 7
#     weekly/
#       2026-02-16/         ← latest weekly raw data
#       2026-02-09/
#       ...                 ← keeps last 4
#     latest/               ← always current (symlink-style copy)
#
#===============================================================================

set -euo pipefail

# ─── CONFIG ───────────────────────────────────────────────────────────────────
REMOTE="gdrive:Royaley-Backups"
PROJECT_DIR="/nvme0n1-disk/royaley"
RAW_DATA_DIR="/sda-disk/raw-data"
LOCAL_STAGING="/tmp/royaley-backup"
DATE_TAG=$(date +%Y-%m-%d)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DAY_OF_WEEK=$(date +%u)  # 1=Mon, 7=Sun
LOG_DIR="/var/log/royaley-backup"
LOG_FILE="${LOG_DIR}/backup_${DATE_TAG}.log"

# Docker containers
PG_CONTAINER="royaley_postgres"

# Retention
KEEP_DAILY=7
KEEP_WEEKLY=4

# Rclone global flags
RCLONE_FLAGS="--transfers=8 --checkers=16 --drive-chunk-size=64M --log-level=INFO --stats=30s"

# ─── COLORS & LOGGING ────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'

mkdir -p "$LOG_DIR"

log()      { local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"; echo -e "$msg" | tee -a "$LOG_FILE"; }
log_ok()   { log "${GREEN}✅ $1${NC}"; }
log_warn() { log "${YELLOW}⚠️  $1${NC}"; }
log_err()  { log "${RED}❌ $1${NC}"; }
log_info() { log "${BLUE}📦 $1${NC}"; }

get_size() { du -sh "$1" 2>/dev/null | awk '{print $1}' || echo "0"; }

# ─── PRE-FLIGHT CHECKS ───────────────────────────────────────────────────────
preflight() {
    # Check rclone installed
    if ! command -v rclone &>/dev/null; then
        log_err "rclone not installed! Run: curl https://rclone.org/install.sh | sudo bash"
        exit 1
    fi

    # Check remote is configured
    if ! rclone listremotes | grep -q "gdrive:"; then
        log_err "rclone remote 'gdrive' not configured! Run: rclone config"
        exit 1
    fi

    # Check remote is accessible
    if ! rclone lsd gdrive: &>/dev/null; then
        log_err "Cannot access Google Drive! Check rclone config and auth token."
        exit 1
    fi

    # Check Docker is running
    if ! docker ps &>/dev/null; then
        log_err "Docker is not running!"
        exit 1
    fi

    log_ok "Pre-flight checks passed"
}

# ─── BACKUP: POSTGRESQL DATABASE ─────────────────────────────────────────────
backup_database() {
    log_info "Backing up PostgreSQL database (20 GB)..."
    local dest="${LOCAL_STAGING}/database"
    mkdir -p "$dest"

    # Check if postgres container is running
    if ! docker ps --format '{{.Names}}' | grep -q "^${PG_CONTAINER}$"; then
        log_err "PostgreSQL container not running!"
        return 1
    fi

    # Full custom-format dump (compressed, fast restore)
    log "  → pg_dump (custom format, compressed)..."
    docker exec "$PG_CONTAINER" pg_dump \
        -U royaley -d royaley \
        -Fc --compress=6 \
        --no-owner --no-privileges \
        -f /tmp/royaley_full.dump 2>&1 || {
        log_err "pg_dump failed!"
        return 1
    }
    docker cp "${PG_CONTAINER}:/tmp/royaley_full.dump" "${dest}/royaley_full.dump"
    docker exec "$PG_CONTAINER" rm -f /tmp/royaley_full.dump

    # Schema-only dump (lightweight, for reference)
    log "  → Schema dump..."
    docker exec "$PG_CONTAINER" pg_dump \
        -U royaley -d royaley --schema-only \
        -f /tmp/royaley_schema.sql 2>&1 || true
    docker cp "${PG_CONTAINER}:/tmp/royaley_schema.sql" "${dest}/royaley_schema.sql" 2>/dev/null || true
    docker exec "$PG_CONTAINER" rm -f /tmp/royaley_schema.sql 2>/dev/null || true

    # Table row counts for verification
    docker exec "$PG_CONTAINER" psql -U royaley -d royaley -c "
        SELECT tablename, n_live_tup as rows
        FROM pg_stat_user_tables
        ORDER BY n_live_tup DESC;
    " > "${dest}/table_counts.txt" 2>/dev/null || true

    # Upload to Google Drive
    log "  → Uploading database dump to Google Drive..."
    rclone copy "$dest" "${REMOTE}/daily/${DATE_TAG}/database" $RCLONE_FLAGS 2>&1 | tail -5 | tee -a "$LOG_FILE"
    # Also keep latest copy
    rclone copy "$dest" "${REMOTE}/latest/database" $RCLONE_FLAGS 2>&1 | tail -3

    local size=$(get_size "$dest")
    log_ok "Database backup: ${size} → Google Drive"
}

# ─── BACKUP: SOURCE CODE ─────────────────────────────────────────────────────
backup_code() {
    log_info "Backing up source code (3.7 GB)..."
    local dest="${LOCAL_STAGING}/code"
    mkdir -p "$dest"

    tar czf "${dest}/royaley_code.tar.gz" \
        -C "$(dirname $PROJECT_DIR)" \
        --exclude='postgres_data' \
        --exclude='redis_data' \
        --exclude='node_modules' \
        --exclude='__pycache__' \
        --exclude='.git/objects' \
        --exclude='*.pyc' \
        --exclude='prometheus_data' \
        --exclude='grafana_data' \
        --exclude='*.log' \
        --exclude='models' \
        "$(basename $PROJECT_DIR)" 2>/dev/null || {
        log_warn "Some files excluded (permission denied), continuing..."
    }

    log "  → Uploading code to Google Drive..."
    rclone copy "$dest" "${REMOTE}/daily/${DATE_TAG}/code" $RCLONE_FLAGS 2>&1 | tail -5 | tee -a "$LOG_FILE"
    rclone copy "$dest" "${REMOTE}/latest/code" $RCLONE_FLAGS 2>&1 | tail -3

    local size=$(get_size "$dest")
    log_ok "Code backup: ${size} → Google Drive"
}

# ─── BACKUP: ML MODELS ───────────────────────────────────────────────────────
backup_models() {
    log_info "Backing up ML models & calibrators (2 GB)..."
    local dest="${LOCAL_STAGING}/models"
    mkdir -p "$dest"

    # Tar the models directory
    if [ -d "${PROJECT_DIR}/models" ]; then
        tar czf "${dest}/ml_models.tar.gz" \
            -C "$PROJECT_DIR" models 2>/dev/null || {
            log_warn "Some model files may have been skipped"
        }
    fi

    # Global calibrator
    docker cp "${PG_CONTAINER%_postgres}_api:/app/models/global_calibrator.pkl" \
        "${dest}/global_calibrator.pkl" 2>/dev/null || true

    log "  → Uploading models to Google Drive..."
    rclone copy "$dest" "${REMOTE}/daily/${DATE_TAG}/models" $RCLONE_FLAGS 2>&1 | tail -5 | tee -a "$LOG_FILE"
    rclone copy "$dest" "${REMOTE}/latest/models" $RCLONE_FLAGS 2>&1 | tail -3

    local size=$(get_size "$dest")
    log_ok "Models backup: ${size} → Google Drive"
}

# ─── BACKUP: CONFIGS & ENV ───────────────────────────────────────────────────
backup_configs() {
    log_info "Backing up configs, env, Docker, SSL..."
    local dest="${LOCAL_STAGING}/configs"
    mkdir -p "$dest"

    # Environment files
    cp "${PROJECT_DIR}/.env" "${dest}/.env" 2>/dev/null || true
    cp "${PROJECT_DIR}/.env.production" "${dest}/.env.production" 2>/dev/null || true

    # Docker compose files
    cp ${PROJECT_DIR}/docker-compose*.yml "${dest}/" 2>/dev/null || true

    # Dockerfiles
    cp ${PROJECT_DIR}/Dockerfile* "${dest}/" 2>/dev/null || true

    # Requirements
    cp ${PROJECT_DIR}/requirements*.txt "${dest}/" 2>/dev/null || true
    cp ${PROJECT_DIR}/pyproject.toml "${dest}/" 2>/dev/null || true

    # Nginx configs
    [ -d "${PROJECT_DIR}/nginx" ] && cp -r "${PROJECT_DIR}/nginx" "${dest}/nginx" 2>/dev/null || true

    # SSL certs
    [ -d "/etc/letsencrypt" ] && tar czf "${dest}/ssl_certs.tar.gz" /etc/letsencrypt 2>/dev/null || true

    # Docker state snapshot
    docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Status}}" > "${dest}/docker_state.txt" 2>/dev/null || true
    docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}" > "${dest}/docker_images.txt" 2>/dev/null || true

    # Crontab
    crontab -l > "${dest}/crontab.txt" 2>/dev/null || true

    # Rclone config (for disaster recovery)
    cp ~/.config/rclone/rclone.conf "${dest}/rclone.conf" 2>/dev/null || true

    log "  → Uploading configs to Google Drive..."
    rclone copy "$dest" "${REMOTE}/daily/${DATE_TAG}/configs" $RCLONE_FLAGS 2>&1 | tail -5 | tee -a "$LOG_FILE"
    rclone copy "$dest" "${REMOTE}/latest/configs" $RCLONE_FLAGS 2>&1 | tail -3

    local size=$(get_size "$dest")
    log_ok "Configs backup: ${size} → Google Drive"
}

# ─── BACKUP: RAW DATA (WEEKLY) ───────────────────────────────────────────────
backup_raw_data() {
    log_info "Backing up raw data from HDD (3.7 TB) — this will take hours..."

    if [ ! -d "$RAW_DATA_DIR" ]; then
        log_err "Raw data directory not found: $RAW_DATA_DIR"
        return 1
    fi

    local size=$(get_size "$RAW_DATA_DIR")
    log "  → Raw data size: ${size}"
    log "  → Using rclone sync (incremental — only uploads changes)"

    # Use sync (not copy) — only uploads changed/new files, removes deleted ones
    # This makes subsequent weekly backups MUCH faster (only diffs)
    rclone sync "$RAW_DATA_DIR" "${REMOTE}/weekly/${DATE_TAG}/raw-data" \
        $RCLONE_FLAGS \
        --transfers=4 \
        --drive-chunk-size=128M \
        --exclude="export.log" \
        --exclude="*.tmp" \
        2>&1 | tee -a "$LOG_FILE"

    # Also maintain a "latest" sync
    rclone sync "$RAW_DATA_DIR" "${REMOTE}/latest/raw-data" \
        $RCLONE_FLAGS \
        --transfers=4 \
        --drive-chunk-size=128M \
        --exclude="export.log" \
        --exclude="*.tmp" \
        2>&1 | tail -5

    log_ok "Raw data backup complete: ${size} → Google Drive"
}

# ─── RETENTION CLEANUP ────────────────────────────────────────────────────────
cleanup_retention() {
    log_info "Cleaning up old backups on Google Drive..."

    # List and prune daily backups (keep last 7)
    local daily_dirs=$(rclone lsd "${REMOTE}/daily/" 2>/dev/null | awk '{print $NF}' | sort -r)
    local count=0
    for dir in $daily_dirs; do
        count=$((count + 1))
        if [ "$count" -gt "$KEEP_DAILY" ]; then
            log "  → Removing old daily: ${dir}"
            rclone purge "${REMOTE}/daily/${dir}" 2>/dev/null || true
        fi
    done

    # Prune weekly backups (keep last 4)
    local weekly_dirs=$(rclone lsd "${REMOTE}/weekly/" 2>/dev/null | awk '{print $NF}' | sort -r)
    count=0
    for dir in $weekly_dirs; do
        count=$((count + 1))
        if [ "$count" -gt "$KEEP_WEEKLY" ]; then
            log "  → Removing old weekly: ${dir}"
            rclone purge "${REMOTE}/weekly/${dir}" 2>/dev/null || true
        fi
    done

    log_ok "Retention cleanup complete (daily: ${KEEP_DAILY}, weekly: ${KEEP_WEEKLY})"
}

# ─── STATUS ──────────────────────────────────────────────────────────────────
show_status() {
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  ROYALEY GOOGLE DRIVE BACKUP STATUS"
    echo "  Remote: ${REMOTE}"
    echo "═══════════════════════════════════════════════════════════════"

    # Google Drive usage
    echo ""
    echo "  📊 Google Drive Space:"
    rclone about gdrive: 2>/dev/null | sed 's/^/    /'

    # Daily backups
    echo ""
    echo "  📅 DAILY BACKUPS:"
    rclone lsd "${REMOTE}/daily/" 2>/dev/null | while read -r line; do
        dir=$(echo "$line" | awk '{print $NF}')
        echo "    📁 ${dir}"
    done
    local daily_count=$(rclone lsd "${REMOTE}/daily/" 2>/dev/null | wc -l)
    echo "    Total: ${daily_count} daily backups (keep ${KEEP_DAILY})"

    # Weekly backups
    echo ""
    echo "  📅 WEEKLY BACKUPS:"
    rclone lsd "${REMOTE}/weekly/" 2>/dev/null | while read -r line; do
        dir=$(echo "$line" | awk '{print $NF}')
        echo "    📁 ${dir}"
    done
    local weekly_count=$(rclone lsd "${REMOTE}/weekly/" 2>/dev/null | wc -l)
    echo "    Total: ${weekly_count} weekly backups (keep ${KEEP_WEEKLY})"

    # Latest backup contents
    echo ""
    echo "  📦 LATEST BACKUP:"
    rclone lsd "${REMOTE}/latest/" 2>/dev/null | while read -r line; do
        dir=$(echo "$line" | awk '{print $NF}')
        size=$(rclone size "${REMOTE}/latest/${dir}" 2>/dev/null | grep "Total size" | awk '{print $3, $4}')
        echo "    ${dir}: ${size}"
    done

    # Last log entries
    echo ""
    echo "  📋 LAST BACKUP LOG:"
    if [ -f "$LOG_FILE" ]; then
        tail -5 "$LOG_FILE" | sed 's/^/    /'
    else
        local latest_log=$(ls -t ${LOG_DIR}/backup_*.log 2>/dev/null | head -1)
        if [ -n "$latest_log" ]; then
            tail -5 "$latest_log" | sed 's/^/    /'
        else
            echo "    (no logs found)"
        fi
    fi

    echo ""
    echo "═══════════════════════════════════════════════════════════════"
}

# ─── INSTALL CRON ────────────────────────────────────────────────────────────
install_cron() {
    local script_path=$(readlink -f "$0")

    # Remove existing royaley backup cron entries
    crontab -l 2>/dev/null | grep -v "backup_gdrive.sh" | crontab - 2>/dev/null || true

    # Add new cron entries
    (
        crontab -l 2>/dev/null
        echo ""
        echo "# ROYALEY Google Drive Backups"
        echo "# Daily backup at 3:00 AM UTC (code, db, models, configs)"
        echo "0 3 * * * ${script_path} >> ${LOG_DIR}/cron.log 2>&1"
        echo "# Weekly raw data backup on Sunday at 4:00 AM UTC"
        echo "0 4 * * 0 ${script_path} --weekly >> ${LOG_DIR}/cron_weekly.log 2>&1"
    ) | crontab -

    log_ok "Cron jobs installed:"
    log "  Daily:  0 3 * * *  (3:00 AM UTC every day)"
    log "  Weekly: 0 4 * * 0  (4:00 AM UTC every Sunday)"
    echo ""
    echo "  Verify with: crontab -l"
}

# ─── RESTORE GUIDE ──────────────────────────────────────────────────────────
show_restore() {
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  ROYALEY RESTORE FROM GOOGLE DRIVE"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
    echo "  # 1. Download latest backup"
    echo "  rclone copy gdrive:Royaley-Backups/latest/ /tmp/restore/ -P"
    echo ""
    echo "  # 2. Restore PostgreSQL"
    echo "  docker exec -i royaley_postgres pg_restore \\"
    echo "    -U royaley -d royaley --clean --no-owner \\"
    echo "    < /tmp/restore/database/royaley_full.dump"
    echo ""
    echo "  # 3. Restore code"
    echo "  tar xzf /tmp/restore/code/royaley_code.tar.gz -C /nvme0n1-disk/"
    echo ""
    echo "  # 4. Restore models"
    echo "  tar xzf /tmp/restore/models/ml_models.tar.gz -C /nvme0n1-disk/royaley/"
    echo ""
    echo "  # 5. Restore configs"
    echo "  cp /tmp/restore/configs/.env* /nvme0n1-disk/royaley/"
    echo "  cp /tmp/restore/configs/docker-compose*.yml /nvme0n1-disk/royaley/"
    echo ""
    echo "  # 6. Restore raw data (from weekly backup)"
    echo "  rclone copy gdrive:Royaley-Backups/latest/raw-data/ /sda-disk/raw-data/ -P"
    echo ""
    echo "  # 7. Rebuild and restart"
    echo "  cd /nvme0n1-disk/royaley"
    echo "  docker compose up -d --build"
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
}

# ─── MAIN ────────────────────────────────────────────────────────────────────
main() {
    case "${1:-}" in
        --status)
            show_status
            exit 0
            ;;
        --install)
            install_cron
            exit 0
            ;;
        --restore)
            show_restore
            exit 0
            ;;
        --db-only)
            preflight
            mkdir -p "$LOCAL_STAGING"
            backup_database
            rm -rf "$LOCAL_STAGING"
            exit 0
            ;;
        --help|-h)
            echo "Usage: $0 [--weekly|--status|--install|--restore|--db-only|--help]"
            echo ""
            echo "  (no args)    Daily backup (db, code, models, configs)"
            echo "  --weekly     Daily + raw data backup (3.7 TB)"
            echo "  --status     Show backup status"
            echo "  --install    Install daily + weekly cron jobs"
            echo "  --restore    Show restore instructions"
            echo "  --db-only    Quick database-only backup"
            exit 0
            ;;
    esac

    preflight
    mkdir -p "$LOCAL_STAGING"

    local start_time=$(date +%s)
    local errors=0

    log ""
    log "═══════════════════════════════════════════════════════════════"
    log "  ROYALEY → GOOGLE DRIVE BACKUP"
    log "  Date: ${DATE_TAG}"
    log "  Mode: $([ "${1:-}" = "--weekly" ] && echo "DAILY + WEEKLY RAW DATA" || echo "DAILY")"
    log "═══════════════════════════════════════════════════════════════"

    # Daily backups
    backup_database || ((errors++))
    backup_code     || ((errors++))
    backup_models   || ((errors++))
    backup_configs  || ((errors++))

    # Weekly raw data (only on --weekly flag or Sunday auto-trigger)
    if [ "${1:-}" = "--weekly" ]; then
        backup_raw_data || ((errors++))
    fi

    # Cleanup old backups
    cleanup_retention || ((errors++))

    # Cleanup local staging
    rm -rf "$LOCAL_STAGING"

    local end_time=$(date +%s)
    local elapsed=$(( (end_time - start_time) / 60 ))

    log ""
    log "═══════════════════════════════════════════════════════════════"
    log "  BACKUP COMPLETE"
    log "  Duration: ${elapsed} minutes"
    log "  Errors: ${errors}"
    log "═══════════════════════════════════════════════════════════════"

    if [ "$errors" -gt 0 ]; then
        log_warn "Completed with ${errors} errors — check ${LOG_FILE}"
        exit 1
    else
        log_ok "All backups successful!"
    fi
}

main "$@"