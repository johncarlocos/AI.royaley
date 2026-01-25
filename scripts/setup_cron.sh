#!/bin/bash
# ROYALEY - Setup Automated Odds Collection
#
# This script sets up cron jobs for:
# 1. Odds collection every 30 minutes
# 2. Closing line capture every 5 minutes (near game times)
# 3. Daily historical import
#
# Usage:
#   chmod +x scripts/setup_cron.sh
#   ./scripts/setup_cron.sh

set -e

echo "=========================================="
echo "ROYALEY - Cron Setup for Odds Collection"
echo "=========================================="

# Create log directory
LOG_DIR="/var/log/royaley"
mkdir -p $LOG_DIR
chmod 755 $LOG_DIR

# Create the cron entries
CRON_FILE="/tmp/royaley_cron"

cat > $CRON_FILE << 'EOF'
# ============================================
# ROYALEY - Automated Odds Collection
# ============================================

# Collect odds from Pinnacle & OddsAPI every 30 minutes
*/30 * * * * cd /nvme0n1-disk/royaley && docker exec royaley_api python scripts/collect_odds.py --sources pinnacle >> /var/log/royaley/pinnacle.log 2>&1

# Capture closing lines every 5 minutes (for games starting soon)
*/5 * * * * cd /nvme0n1-disk/royaley && docker exec royaley_api python scripts/collect_odds.py --closing-lines >> /var/log/royaley/closing_lines.log 2>&1

# Daily historical import at 3 AM (low traffic)
0 3 * * * cd /nvme0n1-disk/royaley && docker exec royaley_api python scripts/import_pinnacle_history.py --all --pages 10 --save >> /var/log/royaley/history_import.log 2>&1

# Weekly full historical sync on Sunday at 4 AM
0 4 * * 0 cd /nvme0n1-disk/royaley && docker exec royaley_api python scripts/import_pinnacle_history.py --all --pages 100 --save >> /var/log/royaley/weekly_history.log 2>&1

# Log rotation - keep 7 days of logs
0 0 * * * find /var/log/royaley -name "*.log" -mtime +7 -delete

EOF

echo "Cron entries to be added:"
echo "-------------------------"
cat $CRON_FILE
echo "-------------------------"

read -p "Install these cron jobs? (y/n) " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Backup existing crontab
    crontab -l > /tmp/crontab_backup_$(date +%Y%m%d) 2>/dev/null || true
    
    # Merge with existing crontab
    (crontab -l 2>/dev/null | grep -v "ROYALEY\|royaley_api\|collect_odds\|import_pinnacle" || true; cat $CRON_FILE) | crontab -
    
    echo ""
    echo "✅ Cron jobs installed!"
    echo ""
    echo "Current crontab:"
    crontab -l
else
    echo "Cancelled. You can manually add these entries with: crontab -e"
fi

rm -f $CRON_FILE

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Logs will be written to:"
echo "  - /var/log/royaley/pinnacle.log"
echo "  - /var/log/royaley/closing_lines.log"
echo "  - /var/log/royaley/history_import.log"
echo ""
echo "To check cron jobs:"
echo "  crontab -l"
echo ""
echo "To remove Royaley cron jobs:"
echo "  crontab -l | grep -v royaley | crontab -"
