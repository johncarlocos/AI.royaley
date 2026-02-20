#!/usr/bin/env python3
"""
Patch predictions_public.py dashboard stats and betting summary
to only count the best pick per game+bet_type.
Run from project root:
  python3 patch_dashboard_stats.py
"""

filepath = 'app/api/routes/predictions_public.py'

with open(filepath, 'r') as f:
    content = f.read()

changes = 0

# ===========================================================================
# We need a reusable subquery for "best prediction IDs"
# Add it after BEST_PICK_FILTER
# ===========================================================================

old_best_pick = '''# Filter to keep only the highest-probability side per game+bet_type
BEST_PICK_FILTER = """
    AND p.id IN (
        SELECT DISTINCT ON (COALESCE(bp.upcoming_game_id, bp.game_id), bp.bet_type)
            bp.id
        FROM predictions bp
        ORDER BY COALESCE(bp.upcoming_game_id, bp.game_id), bp.bet_type, bp.probability DESC
    )
"""'''

new_best_pick = '''# Filter to keep only the highest-probability side per game+bet_type
BEST_PICK_FILTER = """
    AND p.id IN (
        SELECT DISTINCT ON (COALESCE(bp.upcoming_game_id, bp.game_id), bp.bet_type)
            bp.id
        FROM predictions bp
        ORDER BY COALESCE(bp.upcoming_game_id, bp.game_id), bp.bet_type, bp.probability DESC
    )
"""

# Same filter for standalone queries (no alias prefix needed)
BEST_IDS_SUBQUERY = """
    SELECT DISTINCT ON (COALESCE(bp.upcoming_game_id, bp.game_id), bp.bet_type)
        bp.id
    FROM predictions bp
    ORDER BY COALESCE(bp.upcoming_game_id, bp.game_id), bp.bet_type, bp.probability DESC
"""'''

if old_best_pick in content:
    content = content.replace(old_best_pick, new_best_pick)
    changes += 1
    print("✅ 1/7 Added BEST_IDS_SUBQUERY")
else:
    print("⚠️  1/7 BEST_PICK_FILTER not found")

# ===========================================================================
# PATCH 2: total_predictions
# ===========================================================================
old_total = '''total_predictions = (await db.execute(
        text("SELECT COUNT(*) FROM predictions")
    )).scalar() or 0'''

new_total = '''total_predictions = (await db.execute(
        text(f"SELECT COUNT(*) FROM predictions WHERE id IN ({BEST_IDS_SUBQUERY})")
    )).scalar() or 0'''

if old_total in content:
    content = content.replace(old_total, new_total)
    changes += 1
    print("✅ 2/7 Patched total_predictions")
else:
    print("⚠️  2/7 total_predictions not found")

# ===========================================================================
# PATCH 3: tier_a_count
# ===========================================================================
old_tier_a = '''tier_a_count = (await db.execute(
        text("SELECT COUNT(*) FROM predictions WHERE CAST(signal_tier AS TEXT) = 'A'")
    )).scalar() or 0'''

new_tier_a = '''tier_a_count = (await db.execute(
        text(f"SELECT COUNT(*) FROM predictions WHERE CAST(signal_tier AS TEXT) = 'A' AND id IN ({BEST_IDS_SUBQUERY})")
    )).scalar() or 0'''

if old_tier_a in content:
    content = content.replace(old_tier_a, new_tier_a)
    changes += 1
    print("✅ 3/7 Patched tier_a_count")
else:
    print("⚠️  3/7 tier_a_count not found")

# ===========================================================================
# PATCH 4: graded_count, wins, losses, pushes, graded_today, avg_clv, total_pnl
# All these query prediction_results - need to join back to predictions
# ===========================================================================
old_graded_block = '''    graded_count = (await db.execute(
        text("SELECT COUNT(*) FROM prediction_results WHERE actual_result != 'pending'")
    )).scalar() or 0
    pending_count = total_predictions - graded_count

    wins = (await db.execute(
        text("SELECT COUNT(*) FROM prediction_results WHERE actual_result = 'win'")
    )).scalar() or 0
    losses = (await db.execute(
        text("SELECT COUNT(*) FROM prediction_results WHERE actual_result = 'loss'")
    )).scalar() or 0
    pushes = (await db.execute(
        text("SELECT COUNT(*) FROM prediction_results WHERE actual_result = 'push'")
    )).scalar() or 0

    win_rate = round(wins / (wins + losses) * 100, 1) if (wins + losses) > 0 else 0.0

    graded_today = (await db.execute(
        text("SELECT COUNT(*) FROM prediction_results WHERE actual_result != 'pending' AND graded_at::date = CURRENT_DATE")
    )).scalar() or 0

    avg_clv = (await db.execute(
        text("SELECT ROUND(AVG(clv)::numeric, 2) FROM prediction_results WHERE clv IS NOT NULL")
    )).scalar() or 0.0

    total_pnl = (await db.execute(
        text("SELECT ROUND(COALESCE(SUM(profit_loss), 0)::numeric, 2) FROM prediction_results WHERE profit_loss IS NOT NULL")
    )).scalar() or 0.0
    roi = round(float(total_pnl) / max(graded_count * 100, 1) * 100, 1) if graded_count > 0 else 0.0'''

new_graded_block = '''    # Only count best pick per game+bet_type
    _best_pr = f"""
        FROM prediction_results pr2
        JOIN predictions p2 ON pr2.prediction_id = p2.id
        WHERE p2.id IN ({BEST_IDS_SUBQUERY})
    """

    graded_count = (await db.execute(
        text(f"SELECT COUNT(*) {_best_pr} AND pr2.actual_result != 'pending'")
    )).scalar() or 0
    pending_count = total_predictions - graded_count

    wins = (await db.execute(
        text(f"SELECT COUNT(*) {_best_pr} AND pr2.actual_result = 'win'")
    )).scalar() or 0
    losses = (await db.execute(
        text(f"SELECT COUNT(*) {_best_pr} AND pr2.actual_result = 'loss'")
    )).scalar() or 0
    pushes = (await db.execute(
        text(f"SELECT COUNT(*) {_best_pr} AND pr2.actual_result = 'push'")
    )).scalar() or 0

    win_rate = round(wins / (wins + losses) * 100, 1) if (wins + losses) > 0 else 0.0

    graded_today = (await db.execute(
        text(f"SELECT COUNT(*) {_best_pr} AND pr2.actual_result != 'pending' AND pr2.graded_at::date = CURRENT_DATE")
    )).scalar() or 0

    avg_clv = (await db.execute(
        text(f"SELECT ROUND(AVG(pr2.clv)::numeric, 2) {_best_pr} AND pr2.clv IS NOT NULL")
    )).scalar() or 0.0

    total_pnl = (await db.execute(
        text(f"SELECT ROUND(COALESCE(SUM(pr2.profit_loss), 0)::numeric, 2) {_best_pr} AND pr2.profit_loss IS NOT NULL")
    )).scalar() or 0.0
    roi = round(float(total_pnl) / max(graded_count * 100, 1) * 100, 1) if graded_count > 0 else 0.0'''

if old_graded_block in content:
    content = content.replace(old_graded_block, new_graded_block)
    changes += 1
    print("✅ 4/7 Patched grading stats (W/L/P&L/CLV/ROI)")
else:
    print("⚠️  4/7 Grading stats block not found")

# ===========================================================================
# PATCH 5: tier_breakdown
# ===========================================================================
old_tier = '''    tier_rows = (await db.execute(text("""
        SELECT CAST(signal_tier AS TEXT) as tier, COUNT(*) as cnt
        FROM predictions GROUP BY CAST(signal_tier AS TEXT) ORDER BY tier
    """))).fetchall()'''

new_tier = '''    tier_rows = (await db.execute(text(f"""
        SELECT CAST(signal_tier AS TEXT) as tier, COUNT(*) as cnt
        FROM predictions WHERE id IN ({BEST_IDS_SUBQUERY})
        GROUP BY CAST(signal_tier AS TEXT) ORDER BY tier
    """))).fetchall()'''

if old_tier in content:
    content = content.replace(old_tier, new_tier)
    changes += 1
    print("✅ 5/7 Patched tier_breakdown")
else:
    print("⚠️  5/7 tier_breakdown not found")

# ===========================================================================
# PATCH 6: bet_type_breakdown
# ===========================================================================
old_bt = '''    bt_rows = (await db.execute(text("""
        SELECT bet_type, COUNT(*) as cnt
        FROM predictions GROUP BY bet_type ORDER BY cnt DESC
    """))).fetchall()'''

new_bt = '''    bt_rows = (await db.execute(text(f"""
        SELECT bet_type, COUNT(*) as cnt
        FROM predictions WHERE id IN ({BEST_IDS_SUBQUERY})
        GROUP BY bet_type ORDER BY cnt DESC
    """))).fetchall()'''

if old_bt in content:
    content = content.replace(old_bt, new_bt)
    changes += 1
    print("✅ 6/7 Patched bet_type_breakdown")
else:
    print("⚠️  6/7 bet_type_breakdown not found")

# ===========================================================================
# PATCH 7: top_picks - add best pick filter
# ===========================================================================
old_top = '''WHERE COALESCE(ug.scheduled_at, g.scheduled_at) >= NOW() - INTERVAL '2 hours'
        ORDER BY p.probability DESC
        LIMIT 8'''

new_top = '''WHERE COALESCE(ug.scheduled_at, g.scheduled_at) >= NOW() - INTERVAL '2 hours'
        {BEST_PICK_FILTER}
        ORDER BY p.probability DESC
        LIMIT 8'''

if old_top in content:
    content = content.replace(old_top, new_top)
    changes += 1
    print("✅ 7/7 Patched top_picks")
else:
    print("⚠️  7/7 top_picks not found")


# ===========================================================================
# SAVE
# ===========================================================================
if changes > 0:
    with open(filepath, 'w') as f:
        f.write(content)
    print(f"\n✅ Dashboard stats patched ({changes} changes)")
    print(f"  Verify: python3 -c \"import ast; ast.parse(open('{filepath}').read()); print('OK')\"")
else:
    print("\n⚠️  No changes made")