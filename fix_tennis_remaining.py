#!/usr/bin/env python3
"""
ROYALEY - Tennis Patch (fixes the 2 skipped patches from v2)
=============================================================
The v2 patcher couldn't match the _grade_game_predictions signature
and Phase 1 caller because your production file differs slightly.

This script:
  1. Shows you the exact lines so we can diagnose
  2. Applies the fix with very flexible matching
  3. Provides SQL to clean up old ungraded tennis predictions

Usage:
    cd /nvme0n1-disk/royaley
    python3 fix_tennis_remaining.py --show     # Show current code (run this first!)
    python3 fix_tennis_remaining.py --apply    # Apply fixes
    python3 fix_tennis_remaining.py --revert   # Revert
    python3 fix_tennis_remaining.py --cleanup-sql  # Print SQL to clean old tennis
"""

import re
import sys
import shutil
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
SCHEDULER = PROJECT_ROOT / "app" / "pipeline" / "scheduler.py"


def show_current():
    """Show the exact lines around the 2 problem areas."""
    if not SCHEDULER.exists():
        print(f"ERROR: {SCHEDULER} not found")
        sys.exit(1)

    lines = SCHEDULER.read_text().splitlines()

    print("=" * 70)
    print("AREA 1: _grade_game_predictions function signature")
    print("=" * 70)
    for i, line in enumerate(lines):
        if '_grade_game_predictions' in line and 'async def' in line:
            start = max(0, i - 1)
            end = min(len(lines), i + 5)
            for j in range(start, end):
                marker = " >>>" if j == i else "    "
                print(f"  {j+1:4d}{marker} {lines[j]}")
            print()

    print("=" * 70)
    print("AREA 2: All callers of _grade_game_predictions")
    print("=" * 70)
    for i, line in enumerate(lines):
        if '_grade_game_predictions(' in line and 'async def' not in line:
            start = max(0, i - 2)
            end = min(len(lines), i + 2)
            for j in range(start, end):
                marker = " >>>" if j == i else "    "
                print(f"  {j+1:4d}{marker} {lines[j]}")
            print()

    print("=" * 70)
    print("AREA 3: _grade_single function signature")
    print("=" * 70)
    for i, line in enumerate(lines):
        if 'def _grade_single(' in line:
            start = max(0, i - 1)
            end = min(len(lines), i + 8)
            for j in range(start, end):
                marker = " >>>" if j == i else "    "
                print(f"  {j+1:4d}{marker} {lines[j]}")
            print()


def apply_fix():
    """Apply the 2 remaining patches using line-by-line approach."""
    if not SCHEDULER.exists():
        print(f"ERROR: {SCHEDULER} not found")
        sys.exit(1)

    content = SCHEDULER.read_text()
    lines = content.splitlines()
    changes = 0

    # ── Patch 1: Add sport_code to _grade_game_predictions signature ──
    # Find the function definition line and add sport_code parameter
    for i, line in enumerate(lines):
        if 'async def _grade_game_predictions(' in line and 'sport_code' not in line:
            # Replace the closing ) -> int: with , sport_code: str = "") -> int:
            old_line = lines[i]
            new_line = re.sub(
                r'\)\s*->\s*int:',
                ', sport_code: str = "") -> int:',
                old_line
            )
            if new_line != old_line:
                lines[i] = new_line
                print(f"  ✅ Patch 1: Added sport_code to signature (line {i+1})")
                print(f"     Was:  {old_line.strip()}")
                print(f"     Now:  {new_line.strip()}")
                changes += 1
            else:
                print(f"  ⚠️  Could not modify line {i+1}: {old_line.strip()}")
            break
    else:
        if 'sport_code: str = "") -> int:' in content:
            print("  ✅ Patch 1: Already applied (sport_code in signature)")
        else:
            print("  ⚠️  Patch 1: Could not find _grade_game_predictions definition")

    # ── Patch 2: Phase 1 caller — add sport_code=game.sport ──
    # Find: cnt = await _grade_game_predictions(db, game.id, game.home_score, game.away_score)
    # This is the caller inside the "for game in scored_games:" loop
    for i, line in enumerate(lines):
        if ('_grade_game_predictions(db, game.id, game.home_score, game.away_score)' in line
                and 'sport_code' not in line):
            old_line = lines[i]
            new_line = old_line.replace(
                '_grade_game_predictions(db, game.id, game.home_score, game.away_score)',
                '_grade_game_predictions(db, game.id, game.home_score, game.away_score, sport_code=game.sport)'
            )
            if new_line != old_line:
                lines[i] = new_line
                print(f"  ✅ Patch 2: Phase 1 caller now passes sport_code (line {i+1})")
                print(f"     Was:  {old_line.strip()}")
                print(f"     Now:  {new_line.strip()}")
                changes += 1
            break
    else:
        if 'game.away_score, sport_code=game.sport)' in content:
            print("  ✅ Patch 2: Already applied (Phase 1 passes sport_code)")
        else:
            # Try more flexible matching
            for i, line in enumerate(lines):
                if ('_grade_game_predictions(' in line
                        and 'game.home_score' in line
                        and 'game.away_score' in line
                        and 'sport_code' not in line
                        and 'game_uuid' not in line  # Skip Phase 2
                        and 'row.id' not in line):  # Skip manual
                    old_line = lines[i]
                    # Add sport_code before the closing )
                    new_line = re.sub(
                        r'(game\.away_score)\)',
                        r'\1, sport_code=game.sport)',
                        old_line
                    )
                    if new_line != old_line:
                        lines[i] = new_line
                        print(f"  ✅ Patch 2 (flexible): Phase 1 caller fixed (line {i+1})")
                        print(f"     Was:  {old_line.strip()}")
                        print(f"     Now:  {new_line.strip()}")
                        changes += 1
                    break
            else:
                print("  ⚠️  Patch 2: Could not find Phase 1 caller")

    if changes > 0:
        # Backup
        bak = SCHEDULER.with_suffix('.py.bak3')
        if not bak.exists():
            shutil.copy2(SCHEDULER, bak)
            print(f"\n  📦 Backup: {bak}")

        SCHEDULER.write_text('\n'.join(lines) + '\n')
        print(f"\n✅ Applied {changes} patches to {SCHEDULER}")
        print("   Rebuild: docker compose up -d --build api")
    else:
        print("\n✅ No changes needed — all patches already applied!")


def revert():
    bak = SCHEDULER.with_suffix('.py.bak3')
    if bak.exists():
        shutil.copy2(bak, SCHEDULER)
        print(f"  ↩️  Reverted from {bak}")
    else:
        print("  ⚠️  No .bak3 backup found")


def print_cleanup_sql():
    print("""
-- =========================================================
-- ROYALEY: Clean up old ungraded tennis predictions
-- =========================================================
-- Run inside: docker exec royaley_postgres psql -U royaley -d royaley
-- =========================================================

-- 1. Check what we're dealing with
SELECT s.code, ug.status, COUNT(*) as predictions,
       MIN(ug.scheduled_at) as earliest,
       MAX(ug.scheduled_at) as latest
FROM predictions p
JOIN upcoming_games ug ON ug.id = p.upcoming_game_id
JOIN sports s ON s.id = ug.sport_id
LEFT JOIN prediction_results pr ON pr.prediction_id = p.id
WHERE s.code IN ('ATP', 'WTA')
  AND (pr.id IS NULL OR pr.actual_result = 'pending')
GROUP BY s.code, ug.status;

-- 2. Mark old ungraded tennis games as 'expired' (games > 24h past start)
UPDATE upcoming_games
SET status = 'expired', updated_at = NOW()
WHERE id IN (
    SELECT ug.id FROM upcoming_games ug
    JOIN sports s ON s.id = ug.sport_id
    WHERE s.code IN ('ATP', 'WTA')
      AND ug.status = 'scheduled'
      AND ug.scheduled_at < NOW() - INTERVAL '24 hours'
);

-- 3. Void predictions for expired tennis games (mark as 'void' with $0 P&L)
INSERT INTO prediction_results (id, prediction_id, actual_result, profit_loss, graded_at)
SELECT gen_random_uuid(), p.id, 'void', 0.0, NOW()
FROM predictions p
JOIN upcoming_games ug ON ug.id = p.upcoming_game_id
JOIN sports s ON s.id = ug.sport_id
LEFT JOIN prediction_results pr ON pr.prediction_id = p.id
WHERE s.code IN ('ATP', 'WTA')
  AND ug.status = 'expired'
  AND pr.id IS NULL
ON CONFLICT (prediction_id) DO UPDATE SET
    actual_result = 'void',
    profit_loss = 0.0,
    graded_at = NOW();

-- 4. Verify cleanup
SELECT s.code,
       SUM(CASE WHEN pr.actual_result = 'void' THEN 1 ELSE 0 END) as voided,
       SUM(CASE WHEN pr.id IS NULL THEN 1 ELSE 0 END) as still_pending
FROM predictions p
JOIN upcoming_games ug ON ug.id = p.upcoming_game_id
JOIN sports s ON s.id = ug.sport_id
LEFT JOIN prediction_results pr ON pr.prediction_id = p.id
WHERE s.code IN ('ATP', 'WTA')
GROUP BY s.code;
""")


def main():
    parser = argparse.ArgumentParser(description="Fix remaining tennis patches")
    parser.add_argument("--show", action="store_true", help="Show current code around problem areas")
    parser.add_argument("--apply", action="store_true", help="Apply the 2 remaining patches")
    parser.add_argument("--revert", action="store_true", help="Revert from .bak3")
    parser.add_argument("--cleanup-sql", action="store_true", help="Print SQL to void old tennis predictions")
    args = parser.parse_args()

    if args.show:
        show_current()
    elif args.apply:
        apply_fix()
    elif args.revert:
        revert()
    elif args.cleanup_sql:
        print_cleanup_sql()
    else:
        parser.print_help()
        print("\n💡 Start with: python3 fix_tennis_remaining.py --show")


if __name__ == "__main__":
    main()