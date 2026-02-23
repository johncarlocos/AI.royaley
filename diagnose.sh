#!/bin/bash
# ROYALEY Diagnostic Queries
# Run from your server: bash diagnose.sh

echo "============================================"
echo "ROYALEY System Diagnostics"
echo "============================================"

DB_CMD="docker exec royaley_postgres psql -U royaley -d royaley -t -A"

echo ""
echo "1. Predictions: Zero-edge (market fallback) vs ML Model"
echo "--------------------------------------------"
docker exec royaley_postgres psql -U royaley -d royaley -c "
SELECT COUNT(*) as total,
       SUM(CASE WHEN edge = 0 OR edge IS NULL THEN 1 ELSE 0 END) as zero_edge,
       SUM(CASE WHEN edge > 0 THEN 1 ELSE 0 END) as positive_edge
FROM predictions;
"

echo ""
echo "2. Win rate: ML Model vs Market Fallback"
echo "--------------------------------------------"
docker exec royaley_postgres psql -U royaley -d royaley -c "
SELECT 
    CASE WHEN p.edge > 0 THEN 'ML Model' ELSE 'Market Fallback' END as source,
    COUNT(*) as graded,
    SUM(CASE WHEN pr.actual_result = 'win' THEN 1 ELSE 0 END) as wins,
    ROUND(AVG(CASE WHEN pr.actual_result = 'win' THEN 1.0 ELSE 0.0 END) * 100, 1) as win_pct,
    ROUND(SUM(pr.profit_loss)::numeric, 2) as total_pnl
FROM predictions p
JOIN prediction_results pr ON pr.prediction_id = p.id
WHERE pr.actual_result IN ('win', 'loss')
GROUP BY CASE WHEN p.edge > 0 THEN 'ML Model' ELSE 'Market Fallback' END;
"

echo ""
echo "3. Ungraded tennis games (past start time)"
echo "--------------------------------------------"
docker exec royaley_postgres psql -U royaley -d royaley -c "
SELECT s.code, COUNT(*) as ungraded_games
FROM upcoming_games ug
JOIN sports s ON ug.sport_id = s.id
WHERE ug.status = 'scheduled'
  AND ug.scheduled_at < NOW() - INTERVAL '4 hours'
  AND s.code IN ('ATP', 'WTA')
GROUP BY s.code;
"

echo ""
echo "4. Tier accuracy (predicted vs actual)"
echo "--------------------------------------------"
docker exec royaley_postgres psql -U royaley -d royaley -c "
SELECT CAST(p.signal_tier AS TEXT) as tier,
       COUNT(*) as n,
       ROUND(AVG(p.probability)::numeric, 4) as avg_claimed_prob,
       ROUND(AVG(CASE WHEN pr.actual_result = 'win' THEN 1.0 ELSE 0.0 END)::numeric, 4) as actual_win_rate,
       ROUND(AVG(p.probability)::numeric - AVG(CASE WHEN pr.actual_result = 'win' THEN 1.0 ELSE 0.0 END)::numeric, 4) as overconfidence
FROM predictions p
JOIN prediction_results pr ON pr.prediction_id = p.id
WHERE pr.actual_result IN ('win', 'loss')
GROUP BY CAST(p.signal_tier AS TEXT)
ORDER BY tier;
"

echo ""
echo "5. Win rate by sport"
echo "--------------------------------------------"
docker exec royaley_postgres psql -U royaley -d royaley -c "
SELECT s.code as sport,
       COUNT(*) as graded,
       SUM(CASE WHEN pr.actual_result = 'win' THEN 1 ELSE 0 END) as wins,
       ROUND(AVG(CASE WHEN pr.actual_result = 'win' THEN 1.0 ELSE 0.0 END) * 100, 1) as win_pct,
       ROUND(SUM(pr.profit_loss)::numeric, 2) as total_pnl
FROM predictions p
JOIN prediction_results pr ON pr.prediction_id = p.id
JOIN upcoming_games ug ON ug.id = p.upcoming_game_id
JOIN sports s ON s.id = ug.sport_id
WHERE pr.actual_result IN ('win', 'loss')
GROUP BY s.code
ORDER BY graded DESC;
"

echo ""
echo "6. Predictions by bet type"
echo "--------------------------------------------"
docker exec royaley_postgres psql -U royaley -d royaley -c "
SELECT p.bet_type,
       COUNT(*) as total,
       SUM(CASE WHEN pr.actual_result = 'win' THEN 1 ELSE 0 END) as wins,
       SUM(CASE WHEN pr.actual_result = 'loss' THEN 1 ELSE 0 END) as losses,
       ROUND(AVG(CASE WHEN pr.actual_result = 'win' THEN 1.0 ELSE 0.0 END) * 100, 1) as win_pct
FROM predictions p
LEFT JOIN prediction_results pr ON pr.prediction_id = p.id AND pr.actual_result IN ('win', 'loss')
GROUP BY p.bet_type
ORDER BY total DESC;
"

echo ""
echo "7. Tennis tournament keys stored in DB"
echo "--------------------------------------------"
docker exec royaley_postgres psql -U royaley -d royaley -c "
SELECT code, api_key FROM sports WHERE code IN ('ATP', 'WTA');
"

echo ""
echo "8. Sample ungraded tennis predictions (last 10)"
echo "--------------------------------------------"
docker exec royaley_postgres psql -U royaley -d royaley -c "
SELECT s.code, ug.home_team_name, ug.away_team_name,
       ug.scheduled_at, ug.status, ug.home_score, ug.away_score,
       p.bet_type, p.predicted_side
FROM predictions p
JOIN upcoming_games ug ON ug.id = p.upcoming_game_id
JOIN sports s ON s.id = ug.sport_id
LEFT JOIN prediction_results pr ON pr.prediction_id = p.id
WHERE s.code IN ('ATP', 'WTA')
  AND (pr.id IS NULL OR pr.actual_result = 'pending')
ORDER BY ug.scheduled_at DESC
LIMIT 10;
"

echo ""
echo "============================================"
echo "Diagnostics complete"
echo "============================================"