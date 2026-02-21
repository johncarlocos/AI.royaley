#!/usr/bin/env python3
"""
Patch fetch_games.py to add smart routing.
Flips predicted_side for sport+bet_type combos where the model is inverted.

Based on historical data:
  - NBA spread: model picks wrong side (36.8% win → flip to 71.4%)
  - NCAAB total: model always picks under and loses (45.5% → flip to 54.5%)

Run from project root:
  python3 patch_smart_routing.py
"""

filepath = 'app/pipeline/fetch_games.py'

with open(filepath, 'r') as f:
    content = f.read()

changes = 0

# =========================================================================
# PATCH 1: Add FLIP_RULES config near the top of the file
# =========================================================================
old_imports_anchor = 'MIN_ODDS_THRESHOLD'

# Find the line with MIN_ODDS_THRESHOLD and add after it
import re
match = re.search(r'(MIN_ODDS_THRESHOLD\s*=\s*[^\n]+\n)', content)
if match and 'FLIP_RULES' not in content:
    old_line = match.group(1)
    new_line = old_line + """
# ── SMART ROUTING ──
# Flip predicted_side for sport+bet_type combos where the model is historically inverted.
# Key = (sport_code, bet_type), Value = True means flip the side.
# Review and update monthly based on actual performance data.
FLIP_RULES = {
    ("NBA", "spread"): True,     # Model: 36.8% → Flipped: 71.4%
    ("NCAAB", "total"): True,    # Model: 45.5% → Flipped: 54.5%
}

def _flip_side(side: str, bet_type: str) -> str:
    \"\"\"Flip a predicted side to its opposite.\"\"\"
    if bet_type == "spread":
        return "away" if side == "home" else "home"
    elif bet_type == "total":
        return "under" if side == "over" else "over"
    elif bet_type == "moneyline":
        return "away" if side == "home" else "home"
    return side

"""
    content = content.replace(old_line, new_line)
    changes += 1
    print("✅ 1/2 Added FLIP_RULES config and _flip_side helper")
else:
    if 'FLIP_RULES' in content:
        print("⚠️  1/2 FLIP_RULES already exists")
    else:
        print("⚠️  1/2 Could not find MIN_ODDS_THRESHOLD anchor")

# =========================================================================
# PATCH 2: Add flip logic in the prediction loop
# =========================================================================
old_loop = '''        for predicted_side, probability, line_val, odds_val, edge in predictions_to_make:
            # ── MINIMUM ODDS FILTER ──'''

new_loop = '''        for predicted_side, probability, line_val, odds_val, edge in predictions_to_make:
            # ── SMART ROUTING: flip inverted combos ──
            if FLIP_RULES.get((sport_code, bet_type), False):
                original_side = predicted_side
                predicted_side = _flip_side(predicted_side, bet_type)
                probability = 1.0 - probability
                edge = -edge  # Edge flips sign
                # Flip odds and line to match new side
                if bet_type == "spread":
                    line_val = -line_val if line_val else None
                    # Swap to other side's odds from consensus
                    if predicted_side == "home":
                        odds_val = _f(row.pin_home_odds or row.avg_home_odds)
                    else:
                        odds_val = _f(row.pin_away_odds or row.avg_away_odds)
                elif bet_type == "total":
                    if predicted_side == "over":
                        odds_val = _f(row.pin_over_odds or row.avg_over_odds)
                    else:
                        odds_val = _f(row.pin_under_odds or row.avg_under_odds)
                elif bet_type == "moneyline":
                    if predicted_side == "home":
                        odds_val = _f(row.pin_home_ml or row.avg_home_ml)
                    else:
                        odds_val = _f(row.pin_away_ml or row.avg_away_ml)
                logger.info(f"    🔄 Smart flip: {sport_code} {bet_type} {original_side} → {predicted_side}")

            # ── MINIMUM ODDS FILTER ──'''

if old_loop in content:
    content = content.replace(old_loop, new_loop)
    changes += 1
    print("✅ 2/2 Added smart routing flip in prediction loop")
else:
    print("⚠️  2/2 Could not find prediction loop target")

# =========================================================================
# SAVE
# =========================================================================
if changes > 0:
    with open(filepath, 'w') as f:
        f.write(content)
    print(f"\n✅ Smart routing patched ({changes} changes)")
    print(f"  Verify: python3 -c \"import ast; ast.parse(open('{filepath}').read()); print('OK')\"")
else:
    print("\n⚠️  No changes made")