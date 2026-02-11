#!/bin/bash
# ============================================================================
# ROYALEY - Deploy Live Prediction Pipeline
# Run this on the server after git pull
# ============================================================================
set -e

echo "=========================================="
echo "ROYALEY - Deploying Live Pipeline"
echo "=========================================="

cd /nvme0n1-disk/royaley

# Step 1: Run database migration
echo ""
echo "[1/4] Creating upcoming_games and upcoming_odds tables..."
docker exec -i royaley_postgres psql -U royaley -d royaley < migrations/create_upcoming_tables.sql
echo "  ✓ Tables created"

# Step 2: Rebuild API container (new pipeline code)
echo ""
echo "[2/4] Rebuilding API container..."
docker compose build api --no-cache
docker compose up -d api
echo "  ✓ API rebuilt and started"

# Step 3: Wait for API to be healthy
echo ""
echo "[3/4] Waiting for API to be healthy..."
sleep 15
if docker exec royaley_nginx curl -s http://api:8000/api/v1/health | grep -q "healthy"; then
    echo "  ✓ API is healthy"
else
    echo "  ⚠ API may still be starting up..."
fi

# Step 4: Test the pipeline (dry run first)
echo ""
echo "[4/4] Testing pipeline (dry run)..."
docker exec royaley_api python -m app.pipeline.fetch_games --dry-run --sport NBA 2>&1 | head -20

echo ""
echo "=========================================="
echo "Deployment Complete!"
echo ""
echo "Next steps:"
echo "  # Run pipeline for real (fetch games + predictions):"
echo "  docker exec royaley_api python -m app.pipeline.fetch_games --predict"
echo ""
echo "  # Fetch specific sport:"
echo "  docker exec royaley_api python -m app.pipeline.fetch_games --sport NBA --predict"
echo ""
echo "  # Check results:"
echo "  docker exec royaley_postgres psql -U royaley -d royaley -c 'SELECT COUNT(*) FROM upcoming_games;'"
echo "  docker exec royaley_postgres psql -U royaley -d royaley -c 'SELECT COUNT(*) FROM upcoming_odds;'"
echo "  docker exec royaley_postgres psql -U royaley -d royaley -c 'SELECT COUNT(*) FROM predictions;'"
echo "=========================================="
