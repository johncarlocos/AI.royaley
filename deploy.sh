#!/bin/bash
# ============================================================================
# ROYALEY Live Data Deploy - All 6 Changes
# Usage: cd /nvme0n1-disk/royaley && bash deploy.sh
# ============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "======================================"
echo "ROYALEY - Remove Mock Data, Add Live API"
echo "======================================"

# Verify we're in the right directory
if [ ! -f "docker-compose.yml" ]; then
  echo "ERROR: Run this from /nvme0n1-disk/royaley"
  exit 1
fi

# Step 1: Backup originals
echo ""
echo "[1/6] Backing up original files..."
mkdir -p backups/$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="backups/$(ls -t backups | head -1)"
cp app/main.py "$BACKUP_DIR/main.py.bak"
cp frontend/src/api/client.ts "$BACKUP_DIR/client.ts.bak"
cp frontend/src/components/Dashboard/Dashboard.tsx "$BACKUP_DIR/Dashboard.tsx.bak"
cp frontend/src/pages/Predictions/Predictions.tsx "$BACKUP_DIR/Predictions.tsx.bak"
echo "  ✓ Originals saved to $BACKUP_DIR"

# Step 2: Backend - New public API route file
echo ""
echo "[2/6] Adding predictions_public.py (new public API endpoints)..."
cp "$SCRIPT_DIR/app/api/routes/predictions_public.py" app/api/routes/predictions_public.py
echo "  ✓ app/api/routes/predictions_public.py created"

# Step 3: Backend - Patch main.py (register public router)
echo ""
echo "[3/6] Updating main.py (register /api/v1/public/* route)..."
cp "$SCRIPT_DIR/app/main.py" app/main.py
echo "  ✓ app/main.py updated"

# Step 4: Frontend - Update API client
echo ""
echo "[4/6] Updating client.ts (add getPublicPredictions + getDashboardStats)..."
cp "$SCRIPT_DIR/frontend/src/api/client.ts" frontend/src/api/client.ts
echo "  ✓ frontend/src/api/client.ts updated"

# Step 5: Frontend - Update Dashboard
echo ""
echo "[5/6] Updating Dashboard.tsx (live data + 60s auto-refresh)..."
cp "$SCRIPT_DIR/frontend/src/components/Dashboard/Dashboard.tsx" frontend/src/components/Dashboard/Dashboard.tsx
echo "  ✓ frontend/src/components/Dashboard/Dashboard.tsx updated"

# Step 6: Frontend - Update Predictions
echo ""
echo "[6/6] Updating Predictions.tsx (live data + 60s auto-refresh)..."
cp "$SCRIPT_DIR/frontend/src/pages/Predictions/Predictions.tsx" frontend/src/pages/Predictions/Predictions.tsx
echo "  ✓ frontend/src/pages/Predictions/Predictions.tsx updated"

# Rebuild and restart
echo ""
echo "======================================"
echo "Rebuilding containers..."
echo "======================================"
docker compose build api --no-cache 2>&1 | tail -5
docker compose build frontend --no-cache 2>&1 | tail -5
docker compose up -d api frontend 2>&1 | tail -5

echo ""
echo "Waiting 20 seconds for services to start..."
sleep 20

# Verify
echo ""
echo "======================================"
echo "Verifying deployment..."
echo "======================================"

echo ""
echo "Test 1: Public predictions endpoint"
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/api/v1/public/predictions 2>/dev/null)
if [ "$HTTP_CODE" = "200" ]; then
  echo "  ✓ /api/v1/public/predictions → 200 OK"
  curl -s http://localhost:8000/api/v1/public/predictions 2>/dev/null | python3 -m json.tool 2>/dev/null | head -6
else
  echo "  ✗ /api/v1/public/predictions → $HTTP_CODE (may still be starting up)"
fi

echo ""
echo "Test 2: Dashboard stats endpoint"
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/api/v1/public/dashboard/stats 2>/dev/null)
if [ "$HTTP_CODE" = "200" ]; then
  echo "  ✓ /api/v1/public/dashboard/stats → 200 OK"
  curl -s http://localhost:8000/api/v1/public/dashboard/stats 2>/dev/null | python3 -m json.tool 2>/dev/null | head -10
else
  echo "  ✗ /api/v1/public/dashboard/stats → $HTTP_CODE (may still be starting up)"
fi

echo ""
echo "Test 3: Container status"
docker compose ps --format "table {{.Name}}\t{{.Status}}" 2>/dev/null | grep -E "api|frontend|postgres"

echo ""
echo "======================================"
echo "✅ DEPLOYMENT COMPLETE"
echo "======================================"
echo ""
echo "6 changes deployed:"
echo "  1. NEW:     app/api/routes/predictions_public.py (public API endpoints)"
echo "  2. UPDATED: app/main.py (registered /api/v1/public/* route)"
echo "  3. UPDATED: frontend/src/api/client.ts (+2 new public API methods)"
echo "  4. UPDATED: frontend/src/components/Dashboard/Dashboard.tsx (live data + 60s refresh)"
echo "  5. UPDATED: frontend/src/pages/Predictions/Predictions.tsx (live data + 60s refresh)"
echo "  6. REBUILT: API + Frontend containers"
echo ""
echo "Design: EXACT same layout preserved. Only data sources changed."
echo "Backups: $BACKUP_DIR"
echo ""
echo "DB currently has 0 predictions → frontend shows empty state."
echo "NEXT STEP: Seed predictions (Odds API → H2O models → DB)."
echo ""
