#!/usr/bin/env python3
"""
Patch predictions_public.py to only return the best side per game+bet_type.
Run from project root:
  python3 patch_predictions_api.py
"""

filepath = 'app/api/routes/predictions_public.py'

with open(filepath, 'r') as f:
    content = f.read()

changes = 0

# Add a best_pick filter CTE after the existing CTEs
# We'll add a BEST_PICK_FILTER constant after CURRENT_ODDS_CTE definition

old_helper = '''def _safe_int(val):'''

new_helper = '''# Filter to keep only the highest-probability side per game+bet_type
BEST_PICK_FILTER = """
    AND p.id IN (
        SELECT DISTINCT ON (COALESCE(bp.upcoming_game_id, bp.game_id), bp.bet_type)
            bp.id
        FROM predictions bp
        ORDER BY COALESCE(bp.upcoming_game_id, bp.game_id), bp.bet_type, bp.probability DESC
    )
"""


def _safe_int(val):'''

if old_helper in content and 'BEST_PICK_FILTER' not in content:
    content = content.replace(old_helper, new_helper)
    changes += 1
    print("✅ 1/3 Added BEST_PICK_FILTER constant")
else:
    print("⚠️  1/3 BEST_PICK_FILTER already exists or target not found")

# Add filter to COUNT query
old_count = '''WHERE (p.upcoming_game_id IS NOT NULL OR p.game_id IS NOT NULL)
        {where_sql}
    """), params)
    total = count_result.scalar() or 0'''

new_count = '''WHERE (p.upcoming_game_id IS NOT NULL OR p.game_id IS NOT NULL)
        {BEST_PICK_FILTER}
        {where_sql}
    """), params)
    total = count_result.scalar() or 0'''

if old_count in content:
    content = content.replace(old_count, new_count)
    changes += 1
    print("✅ 2/3 Added filter to COUNT query")
else:
    print("⚠️  2/3 COUNT query target not found")

# Add filter to main data query
old_data_where = '''WHERE (p.upcoming_game_id IS NOT NULL OR p.game_id IS NOT NULL)
        {where_sql}
        ORDER BY game_time ASC'''

new_data_where = '''WHERE (p.upcoming_game_id IS NOT NULL OR p.game_id IS NOT NULL)
        {BEST_PICK_FILTER}
        {where_sql}
        ORDER BY game_time ASC'''

if old_data_where in content:
    content = content.replace(old_data_where, new_data_where)
    changes += 1
    print("✅ 3/3 Added filter to data query")
else:
    print("⚠️  3/3 Data query target not found")

if changes > 0:
    with open(filepath, 'w') as f:
        f.write(content)
    print(f"\n✅ predictions_public.py patched ({changes} changes)")
    print(f"  Verify: python3 -c \"import ast; ast.parse(open('{filepath}').read()); print('OK')\"")
else:
    print("\n⚠️  No changes made")