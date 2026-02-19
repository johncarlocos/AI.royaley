#!/bin/bash
#===============================================================================
# ROYALEY AUTOMATED BACKUP SYSTEM
#===============================================================================
#
# Backs up all critical data to 16TB HDD (/sda-disk/backup/)
#
# What gets backed up:
#   1. PostgreSQL database (full pg_dump)
#   2. Application code & configs
#   3. ML trained models & calibration data
#   4. Environment files & secrets
#   5. Docker compose configs
#   6. Nginx & SSL configs
#
# Retention: Keeps last 7 daily + last 4 weekly + last 3 monthly backups
#
# Usage:
#   ./backup.sh              # Run full backup
#   ./backup.sh --status     # Show backup status
#   ./backup.sh --restore    # List available restore points
#   ./backup.sh --install    # Install daily cron job
#
#===============================================================================

set -euo pipefail

# ─── CONFIG ───────────────────────────────────────────────────────────────────
BACKUP_ROOT="/sda-disk/backup"
PROJECT_DIR="/nvme0n1-disk/royaley"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DATE_TAG=$(date +%Y%m%d)
DAY_OF_WEEK=$(date +%u)  # 1=Monday, 7=Sunday
DAY_OF_MONTH=$(date +%d)
LOG_FILE="${BACKUP_ROOT}/backup.log"

# Retention policy
KEEP_DAILY=7
KEEP_WEEKLY=4
KEEP_MONTHLY=3

# Docker container names
PG_CONTAINER="royaley_postgres"
API_CONTAINER="royaley_api"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# ─── FUNCTIONS ────────────────────────────────────────────────────────────────

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo -e "$msg"
    echo "$msg" >> "$LOG_FILE" 2>/dev/null || true
}

log_ok()    { log "${GREEN}✅ $1${NC}"; }
log_warn()  { log "${YELLOW}⚠️  $1${NC}"; }
log_err()   { log "${RED}❌ $1${NC}"; }
log_info()  { log "${BLUE}📦 $1${NC}"; }

get_size() {
    du -sh "$1" 2>/dev/null | awk '{print $1}' || echo "0"
}

# ─── BACKUP FUNCTIONS ────────────────────────────────────────────────────────

backup_postgres() {
    log_info "Backing up PostgreSQL database..."
    local dest="${BACKUP_DIR}/postgres"
    mkdir -p "$dest"

    # Check if postgres container is running
    if ! docker ps --format '{{.Names}}' | grep -q "^${PG_CONTAINER}$"; then
        log_err "PostgreSQL container not running!"
        return 1
    fi

    # Full database dump (custom format for fast restore)
    log "  → Full dump (custom format)..."
    docker exec "$PG_CONTAINER" pg_dump \
        -U royaley \
        -d royaley \
        -Fc \
        --no-owner \
        --no-privileges \
        -f /tmp/royaley_full.dump 2>&1 || {
        log_err "pg_dump failed!"
        return 1
    }

    # Copy dump out of container
    docker cp "${PG_CONTAINER}:/tmp/royaley_full.dump" "${dest}/royaley_full.dump"
    docker exec "$PG_CONTAINER" rm -f /tmp/royaley_full.dump

    # Also do a plain SQL dump (human-readable, can grep)
    log "  → Schema-only dump..."
    docker exec "$PG_CONTAINER" pg_dump \
        -U royaley \
        -d royaley \
        --schema-only \
        -f /tmp/royaley_schema.sql 2>&1 || true
    docker cp "${PG_CONTAINER}:/tmp/royaley_schema.sql" "${dest}/royaley_schema.sql" 2>/dev/null || true
    docker exec "$PG_CONTAINER" rm -f /tmp/royaley_schema.sql 2>/dev/null || true

    # Dump table row counts for verification
    docker exec "$PG_CONTAINER" psql -U royaley -d royaley -c "
        SELECT schemaname, tablename, n_live_tup as row_count
        FROM pg_stat_user_tables
        ORDER BY n_live_tup DESC;
    " > "${dest}/table_counts.txt" 2>/dev/null || true

    local size=$(get_size "$dest")
    log_ok "PostgreSQL backup: ${size}"
}

backup_code() {
    log_info "Backing up application code..."
    local dest="${BACKUP_DIR}/code"
    mkdir -p "$dest"

    # Tar the entire project (excluding large/generated dirs)
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
        --exclude='raw-data' \
        "$(basename $PROJECT_DIR)" 2>/dev/null || {
        log_warn "Some files excluded (permission denied)"
    }

    local size=$(get_size "$dest")
    log_ok "Code backup: ${size}"
}

backup_env_configs() {
    log_info "Backing up environment & configs..."
    local dest="${BACKUP_DIR}/configs"
    mkdir -p "$dest"

    # Environment files
    cp "${PROJECT_DIR}/.env" "${dest}/.env" 2>/dev/null || true
    cp "${PROJECT_DIR}/.env.production" "${dest}/.env.production" 2>/dev/null || true

    # Docker compose files
    cp ${PROJECT_DIR}/docker-compose*.yml "${dest}/" 2>/dev/null || true

    # Nginx configs
    if [ -d "${PROJECT_DIR}/nginx" ]; then
        cp -r "${PROJECT_DIR}/nginx" "${dest}/nginx" 2>/dev/null || true
    fi

    # SSL certs (if any)
    if [ -d "/etc/letsencrypt" ]; then
        tar czf "${dest}/ssl_certs.tar.gz" /etc/letsencrypt 2>/dev/null || true
    fi

    # Dockerfile(s)
    cp ${PROJECT_DIR}/Dockerfile* "${dest}/" 2>/dev/null || true

    # Requirements / package files
    cp ${PROJECT_DIR}/requirements*.txt "${dest}/" 2>/dev/null || true
    cp ${PROJECT_DIR}/pyproject.toml "${dest}/" 2>/dev/null || true
    cp ${PROJECT_DIR}/package*.json "${dest}/" 2>/dev/null || true

    local size=$(get_size "$dest")
    log_ok "Config backup: ${size}"
}

backup_models() {
    log_info "Backing up ML models & calibration data..."
    local dest="${BACKUP_DIR}/models"
    mkdir -p "$dest"

    # H2O models
    if [ -d "${PROJECT_DIR}/models" ]; then
        tar czf "${dest}/ml_models.tar.gz" \
            -C "$PROJECT_DIR" models 2>/dev/null || {
            log_warn "Some model files may be missing"
        }
    fi

    # Also check inside docker containers for models
    for dir in /app/models /app/data/models /app/ml_models; do
        docker cp "${API_CONTAINER}:${dir}" "${dest}/" 2>/dev/null || true
    done

    # Calibration data
    if [ -d "${PROJECT_DIR}/calibration" ]; then
        cp -r "${PROJECT_DIR}/calibration" "${dest}/calibration" 2>/dev/null || true
    fi
    docker cp "${API_CONTAINER}:/app/calibration" "${dest}/calibration_docker" 2>/dev/null || true

    # Prediction data / historical results
    if [ -d "${PROJECT_DIR}/data" ]; then
        tar czf "${dest}/data_dir.tar.gz" \
            -C "$PROJECT_DIR" data \
            --exclude='*.csv.gz' \
            --exclude='raw' 2>/dev/null || true
    fi

    local size=$(get_size "$dest")
    log_ok "Models backup: ${size}"
}

backup_docker_state() {
    log_info "Backing up Docker state info..."
    local dest="${BACKUP_DIR}/docker_state"
    mkdir -p "$dest"

    # Running containers
    docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}" \
        > "${dest}/running_containers.txt" 2>/dev/null || true

    # Docker images list
    docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}" \
        > "${dest}/images.txt" 2>/dev/null || true

    # Docker volumes
    docker volume ls > "${dest}/volumes.txt" 2>/dev/null || true

    # Docker networks
    docker network ls > "${dest}/networks.txt" 2>/dev/null || true

    # Disk usage
    df -h > "${dest}/disk_usage.txt" 2>/dev/null || true
    docker system df > "${dest}/docker_disk.txt" 2>/dev/null || true

    log_ok "Docker state saved"
}

# ─── RETENTION MANAGEMENT ────────────────────────────────────────────────────

manage_retention() {
    log_info "Managing backup retention..."

    local daily_dir="${BACKUP_ROOT}/daily"
    local weekly_dir="${BACKUP_ROOT}/weekly"
    local monthly_dir="${BACKUP_ROOT}/monthly"

    mkdir -p "$daily_dir" "$weekly_dir" "$monthly_dir"

    # Move current backup to daily
    if [ -d "$BACKUP_DIR" ]; then
        mv "$BACKUP_DIR" "${daily_dir}/backup_${TIMESTAMP}"
    fi

    # Copy to weekly (every Sunday)
    if [ "$DAY_OF_WEEK" = "7" ]; then
        log "  → Creating weekly backup..."
        cp -al "${daily_dir}/backup_${TIMESTAMP}" "${weekly_dir}/backup_${TIMESTAMP}" 2>/dev/null || \
        cp -r "${daily_dir}/backup_${TIMESTAMP}" "${weekly_dir}/backup_${TIMESTAMP}" 2>/dev/null || true
    fi

    # Copy to monthly (1st of month)
    if [ "$DAY_OF_MONTH" = "01" ]; then
        log "  → Creating monthly backup..."
        cp -al "${daily_dir}/backup_${TIMESTAMP}" "${monthly_dir}/backup_${TIMESTAMP}" 2>/dev/null || \
        cp -r "${daily_dir}/backup_${TIMESTAMP}" "${monthly_dir}/backup_${TIMESTAMP}" 2>/dev/null || true
    fi

    # Prune old daily backups
    local count=$(ls -dt ${daily_dir}/backup_* 2>/dev/null | wc -l)
    if [ "$count" -gt "$KEEP_DAILY" ]; then
        ls -dt ${daily_dir}/backup_* | tail -n +$((KEEP_DAILY + 1)) | while read dir; do
            log "  → Removing old daily: $(basename $dir)"
            rm -rf "$dir"
        done
    fi

    # Prune old weekly backups
    count=$(ls -dt ${weekly_dir}/backup_* 2>/dev/null | wc -l)
    if [ "$count" -gt "$KEEP_WEEKLY" ]; then
        ls -dt ${weekly_dir}/backup_* | tail -n +$((KEEP_WEEKLY + 1)) | while read dir; do
            log "  → Removing old weekly: $(basename $dir)"
            rm -rf "$dir"
        done
    fi

    # Prune old monthly backups
    count=$(ls -dt ${monthly_dir}/backup_* 2>/dev/null | wc -l)
    if [ "$count" -gt "$KEEP_MONTHLY" ]; then
        ls -dt ${monthly_dir}/backup_* | tail -n +$((KEEP_MONTHLY + 1)) | while read dir; do
            log "  → Removing old monthly: $(basename $dir)"
            rm -rf "$dir"
        done
    fi

    log_ok "Retention policy applied (${KEEP_DAILY}d / ${KEEP_WEEKLY}w / ${KEEP_MONTHLY}m)"
}

# ─── STATUS ──────────────────────────────────────────────────────────────────

show_status() {
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  ROYALEY BACKUP STATUS"
    echo "  Location: ${BACKUP_ROOT}"
    echo "═══════════════════════════════════════════════════════════════"

    for period in daily weekly monthly; do
        local dir="${BACKUP_ROOT}/${period}"
        local count=$(ls -d ${dir}/backup_* 2>/dev/null | wc -l)
        local size=$(du -sh "$dir" 2>/dev/null | awk '{print $1}' || echo "0")

        echo ""
        echo "  📁 ${period^^} (${count} backups, ${size} total)"
        echo "  ─────────────────────────────────────────────────"

        if [ "$count" -gt 0 ]; then
            ls -dt ${dir}/backup_* 2>/dev/null | head -5 | while read bdir; do
                local bname=$(basename "$bdir")
                local bsize=$(du -sh "$bdir" 2>/dev/null | awk '{print $1}')
                local bdate=$(echo "$bname" | sed 's/backup_//' | sed 's/_/ /')

                # Check what's inside
                local has_pg="—"
                local has_code="—"
                local has_models="—"
                [ -f "${bdir}/postgres/royaley_full.dump" ] && has_pg="✅"
                [ -f "${bdir}/code/royaley_code.tar.gz" ] && has_code="✅"
                [ -d "${bdir}/models" ] && has_models="✅"

                echo "    ${bdate}  ${bsize}  DB:${has_pg} Code:${has_code} Models:${has_models}"
            done
        else
            echo "    (none)"
        fi
    done

    echo ""
    # HDD usage
    local usage=$(df -h /sda-disk 2>/dev/null | tail -1)
    echo "  💿 HDD: ${usage}"
    echo ""

    # Last backup log
    if [ -f "$LOG_FILE" ]; then
        echo "  📋 Last backup log entry:"
        tail -3 "$LOG_FILE" | sed 's/^/    /'
    fi

    echo "═══════════════════════════════════════════════════════════════"
}

# ─── RESTORE HELPER ──────────────────────────────────────────────────────────

show_restore() {
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  AVAILABLE RESTORE POINTS"
    echo "═══════════════════════════════════════════════════════════════"

    for period in daily weekly monthly; do
        local dir="${BACKUP_ROOT}/${period}"
        echo ""
        echo "  ${period^^}:"
        ls -dt ${dir}/backup_* 2>/dev/null | while read bdir; do
            local bname=$(basename "$bdir")
            local bsize=$(du -sh "$bdir" 2>/dev/null | awk '{print $1}')
            echo "    ${bname}  (${bsize})"
        done
    done

    echo ""
    echo "  RESTORE COMMANDS:"
    echo "  ─────────────────────────────────────────────────────────"
    echo ""
    echo "  # Restore PostgreSQL:"
    echo "  docker exec -i royaley_postgres pg_restore \\"
    echo "    -U royaley -d royaley --clean --no-owner \\"
    echo "    < /sda-disk/backup/daily/backup_XXXXX/postgres/royaley_full.dump"
    echo ""
    echo "  # Restore code:"
    echo "  tar xzf /sda-disk/backup/daily/backup_XXXXX/code/royaley_code.tar.gz \\"
    echo "    -C /nvme0n1-disk/"
    echo ""
    echo "  # Restore configs:"
    echo "  cp /sda-disk/backup/daily/backup_XXXXX/configs/.env* /nvme0n1-disk/royaley/"
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
}

# ─── INSTALL CRON ────────────────────────────────────────────────────────────

install_cron() {
    local script_path=$(readlink -f "$0")
    local cron_entry="0 3 * * * ${script_path} >> ${BACKUP_ROOT}/cron.log 2>&1"

    # Check if already installed
    if crontab -l 2>/dev/null | grep -q "backup.sh"; then
        log_warn "Cron job already exists:"
        crontab -l | grep "backup.sh"
        return
    fi

    # Add to crontab
    (crontab -l 2>/dev/null; echo "$cron_entry") | crontab -
    log_ok "Cron job installed: Daily at 3:00 AM UTC"
    log "  Entry: ${cron_entry}"
    echo ""
    echo "  Verify with: crontab -l"
}

# ─── MAIN ────────────────────────────────────────────────────────────────────

main() {
    case "${1:-}" in
        --status)
            show_status
            exit 0
            ;;
        --restore)
            show_restore
            exit 0
            ;;
        --install)
            install_cron
            exit 0
            ;;
        --help|-h)
            echo "Usage: $0 [--status|--restore|--install|--help]"
            echo ""
            echo "  (no args)   Run full backup"
            echo "  --status    Show backup status"
            echo "  --restore   List restore points & commands"
            echo "  --install   Install daily cron job (3 AM UTC)"
            exit 0
            ;;
    esac

    # Create backup directory
    BACKUP_DIR="${BACKUP_ROOT}/current_${TIMESTAMP}"
    mkdir -p "$BACKUP_DIR" "$BACKUP_ROOT"

    log "═══════════════════════════════════════════════════════════════"
    log "  ROYALEY BACKUP STARTING"
    log "  Time: $(date)"
    log "  Target: ${BACKUP_DIR}"
    log "═══════════════════════════════════════════════════════════════"

    local start_time=$(date +%s)
    local errors=0

    # 1. PostgreSQL (most critical)
    backup_postgres || ((errors++))

    # 2. Application code
    backup_code || ((errors++))

    # 3. Environment & configs
    backup_env_configs || ((errors++))

    # 4. ML models & calibration
    backup_models || ((errors++))

    # 5. Docker state
    backup_docker_state || ((errors++))

    # 6. Manage retention
    manage_retention || ((errors++))

    local end_time=$(date +%s)
    local elapsed=$((end_time - start_time))
    local total_size=$(du -sh "${BACKUP_ROOT}" 2>/dev/null | awk '{print $1}')

    log ""
    log "═══════════════════════════════════════════════════════════════"
    log "  BACKUP COMPLETE"
    log "  Duration: ${elapsed}s"
    log "  Total backup size: ${total_size}"
    log "  Errors: ${errors}"
    log "═══════════════════════════════════════════════════════════════"

    if [ "$errors" -gt 0 ]; then
        log_warn "Completed with ${errors} errors - check log"
        exit 1
    else
        log_ok "All backups successful!"
        exit 0
    fi
}

main "$@"