#!/usr/bin/env python3
"""
Run this on the server:
  cd /nvme0n1-disk/royaley
  python3 patch_scheduler.py
"""
import sys

filepath = 'app/services/scheduling/scheduler_service.py'

with open(filepath, 'r') as f:
    content = f.read()

changes = 0

# 1. Replace SQL query fields to include edge, line, tier, scheduled_at
old_sql = """p.confidence, p.odds_at_prediction,
                           ug.home_team_name, ug.away_team_name"""
new_sql = """p.probability as confidence, p.odds_at_prediction,
                           p.edge, p.line_at_prediction, p.signal_tier,
                           ug.home_team_name, ug.away_team_name,
                           ug.scheduled_at"""

if old_sql in content:
    content = content.replace(old_sql, new_sql)
    changes += 1
    print("✅ 1/3 SQL query updated")
else:
    print("⚠️  1/3 SQL query already patched or not found")

# 2. Replace totals team display (show Over/Under instead of "X vs Y")
old_totals = '''elif "over" in side.lower() or "under" in side.lower():
                        team = f"{r.home_team_name} vs {r.away_team_name}"'''
new_totals = '''elif "over" in side.lower() or "under" in side.lower():
                        team = side.capitalize()'''

if old_totals in content:
    content = content.replace(old_totals, new_totals)
    changes += 1
    print("✅ 2/3 Totals display updated")
else:
    print("⚠️  2/3 Totals display already patched or not found")

# 3. Replace preds.append block with richer data
old_append = '''preds.append({
                        "sport": r.sport,
                        "bet_type": r.bet_type,
                        "team": team,
                        "confidence": float(r.confidence) if r.confidence else 0.5,
                        "odds": r.odds_at_prediction or "N/A",
                    })'''
new_append = '''# Format game time in PST
                    game_time = ""
                    if r.scheduled_at:
                        from datetime import timezone, timedelta
                        pst = timezone(timedelta(hours=-8))
                        game_dt = r.scheduled_at.replace(tzinfo=timezone.utc).astimezone(pst)
                        game_time = game_dt.strftime("%I:%M %p PST")

                    preds.append({
                        "sport": r.sport,
                        "bet_type": r.bet_type,
                        "predicted_side": r.predicted_side,
                        "team": team,
                        "home": r.home_team_name,
                        "away": r.away_team_name,
                        "confidence": float(r.confidence) if r.confidence else 0.5,
                        "odds": r.odds_at_prediction or "N/A",
                        "edge": float(r.edge) if r.edge else 0,
                        "line": r.line_at_prediction or "",
                        "tier": r.signal_tier or "",
                        "game_time": game_time,
                    })'''

if old_append in content:
    content = content.replace(old_append, new_append)
    changes += 1
    print("✅ 3/3 Prediction data enriched")
else:
    print("⚠️  3/3 Prediction data already patched or not found")

if changes > 0:
    with open(filepath, 'w') as f:
        f.write(content)
    print(f"\n✅ Scheduler patched ({changes} changes). Verify:")
    print(f"  python3 -c \"import ast; ast.parse(open('{filepath}').read()); print('Syntax OK')\"")
else:
    print("\n⚠️  No changes made - file may already be patched")