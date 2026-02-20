#!/usr/bin/env python3
"""
Patch scheduler.py to return richer grading stats.
Run from project root:
  cd /nvme0n1-disk/royaley
  python3 patch_grading.py
"""

filepath = 'app/pipeline/scheduler.py'

with open(filepath, 'r') as f:
    content = f.read()

changes = 0

# =============================================================================
# PATCH 1: Change _grade_game_predictions to return detailed results
# =============================================================================

old_grade_return = '''async def _grade_game_predictions(db: AsyncSession, game_id, home_score: int, away_score: int) -> int:
    """
    Grade all predictions for a single game that already has scores.
    Returns count of predictions graded.
    """'''

new_grade_return = '''async def _grade_game_predictions(db: AsyncSession, game_id, home_score: int, away_score: int) -> dict:
    """
    Grade all predictions for a single game that already has scores.
    Returns dict with count, wins, losses, pushes, total_pnl.
    """'''

if old_grade_return in content:
    content = content.replace(old_grade_return, new_grade_return)
    changes += 1
    print("✅ 1/4 Grade function signature updated")
else:
    print("⚠️  1/4 Grade function signature not found or already patched")

# =============================================================================
# PATCH 2: Track W/L/P&L inside _grade_game_predictions
# =============================================================================

old_count_init = '''    preds = await db.execute(text("""
        SELECT p.id, p.bet_type, p.predicted_side,
               p.line_at_prediction, p.odds_at_prediction,
               p.home_line_open, p.away_line_open, p.total_open,
               p.home_ml_open, p.away_ml_open,
               pr.closing_line, pr.closing_odds
        FROM predictions p
        LEFT JOIN prediction_results pr ON pr.prediction_id = p.id
        WHERE p.upcoming_game_id = :gid
    """), {"gid": game_id})
    count = 0'''

new_count_init = '''    preds = await db.execute(text("""
        SELECT p.id, p.bet_type, p.predicted_side,
               p.line_at_prediction, p.odds_at_prediction,
               p.home_line_open, p.away_line_open, p.total_open,
               p.home_ml_open, p.away_ml_open,
               pr.closing_line, pr.closing_odds
        FROM predictions p
        LEFT JOIN prediction_results pr ON pr.prediction_id = p.id
        WHERE p.upcoming_game_id = :gid
    """), {"gid": game_id})
    count = 0
    result_stats = {"wins": 0, "losses": 0, "pushes": 0, "pnl": 0.0, "details": []}'''

if old_count_init in content:
    content = content.replace(old_count_init, new_count_init)
    changes += 1
    print("✅ 2/4 Result stats tracking added")
else:
    print("⚠️  2/4 Result stats tracking not found or already patched")

# =============================================================================
# PATCH 3: Track each graded prediction result
# =============================================================================

old_count_inc = '''            count += 1
            logger.info(f"      {pred.bet_type} {pred.predicted_side}: {result_val} (P/L: {pnl})")'''

new_count_inc = '''            count += 1
            # Track W/L/P&L
            if result_val == "win":
                result_stats["wins"] += 1
            elif result_val == "loss":
                result_stats["losses"] += 1
            else:
                result_stats["pushes"] += 1
            result_stats["pnl"] += float(pnl) if pnl else 0
            result_stats["details"].append({
                "bet_type": pred.bet_type,
                "side": pred.predicted_side,
                "result": result_val,
                "pnl": float(pnl) if pnl else 0,
            })
            logger.info(f"      {pred.bet_type} {pred.predicted_side}: {result_val} (P/L: {pnl})")'''

if old_count_inc in content:
    content = content.replace(old_count_inc, new_count_inc)
    changes += 1
    print("✅ 3/4 W/L/P&L tracking per prediction")
else:
    print("⚠️  3/4 W/L/P&L tracking not found or already patched")

# =============================================================================
# PATCH 4: Return rich dict instead of count, and enrich stats in Phase 1 loop
# =============================================================================

old_return = '''    return count'''

# Only replace the FIRST occurrence (inside _grade_game_predictions)
if old_return in content:
    content = content.replace(old_return, '''    result_stats["count"] = count
    return result_stats''', 1)
    changes += 1
    print("✅ 4/4 Return rich result dict")
else:
    print("⚠️  4/4 Return statement not found or already patched")

# =============================================================================
# PATCH 5: Update Phase 1 loop to use rich result and build game_results
# =============================================================================

old_phase1_loop = '''    if scored_games:
        logger.info(f"  Phase 1: {len(scored_games)} games with scores but ungraded predictions")
        for game in scored_games:
            logger.info(f"    Grading: {game.sport} {game.home_team_name} {game.home_score}-{game.away_score} {game.away_team_name}")
            cnt = await _grade_game_predictions(db, game.id, game.home_score, game.away_score)
            if cnt > 0:
                stats["games_graded"] += 1
                stats["predictions_graded"] += cnt
                stats["already_scored"] += 1'''

new_phase1_loop = '''    stats["game_results"] = []
    stats["wins"] = 0
    stats["losses"] = 0
    stats["pushes"] = 0
    stats["total_pnl"] = 0.0

    if scored_games:
        logger.info(f"  Phase 1: {len(scored_games)} games with scores but ungraded predictions")
        for game in scored_games:
            logger.info(f"    Grading: {game.sport} {game.home_team_name} {game.home_score}-{game.away_score} {game.away_team_name}")
            grade_result = await _grade_game_predictions(db, game.id, game.home_score, game.away_score)
            cnt = grade_result.get("count", 0) if isinstance(grade_result, dict) else grade_result
            if cnt > 0:
                stats["games_graded"] += 1
                stats["predictions_graded"] += cnt
                stats["already_scored"] += 1
                if isinstance(grade_result, dict):
                    stats["wins"] += grade_result.get("wins", 0)
                    stats["losses"] += grade_result.get("losses", 0)
                    stats["pushes"] += grade_result.get("pushes", 0)
                    stats["total_pnl"] += grade_result.get("pnl", 0)
                    stats["game_results"].append({
                        "sport": game.sport,
                        "home": game.home_team_name,
                        "away": game.away_team_name,
                        "home_score": game.home_score,
                        "away_score": game.away_score,
                        "wins": grade_result.get("wins", 0),
                        "losses": grade_result.get("losses", 0),
                        "pnl": grade_result.get("pnl", 0),
                    })'''

if old_phase1_loop in content:
    content = content.replace(old_phase1_loop, new_phase1_loop)
    changes += 1
    print("✅ 5/5 Phase 1 loop enriched with game results")
else:
    print("⚠️  5/5 Phase 1 loop not found or already patched")

# =============================================================================
# SAVE
# =============================================================================
if changes > 0:
    with open(filepath, 'w') as f:
        f.write(content)
    print(f"\n✅ scheduler.py patched ({changes} changes)")
    print(f"  Verify: python3 -c \"import ast; ast.parse(open('{filepath}').read()); print('OK')\"")
else:
    print("\n⚠️  No changes made")