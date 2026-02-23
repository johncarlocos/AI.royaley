#!/usr/bin/env python3
"""
ROYALEY - Bug Fix Patcher
Applies all critical fixes to the deployed codebase.

Usage:
    cd /nvme0n1-disk/royaley
    python3 apply_fixes.py --dry-run    # Preview changes (safe)
    python3 apply_fixes.py              # Apply changes
    python3 apply_fixes.py --revert     # Revert from backups

Creates .bak backup of every file before modifying.
"""

import os
import sys
import shutil
import argparse
from pathlib import Path

# Auto-detect project root
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR  # assumes script is in project root

# Files to patch
FILES = {
    "scheduler": PROJECT_ROOT / "app" / "pipeline" / "scheduler.py",
    "model_loader": PROJECT_ROOT / "app" / "pipeline" / "model_loader.py",
    "fetch_games": PROJECT_ROOT / "app" / "pipeline" / "fetch_games.py",
    "calibrator": PROJECT_ROOT / "app" / "pipeline" / "calibrator.py",
}

patches_applied = 0
patches_skipped = 0


def apply_patch(filepath: Path, old_text: str, new_text: str, description: str, dry_run: bool = False):
    """Replace old_text with new_text in filepath."""
    global patches_applied, patches_skipped

    if not filepath.exists():
        print(f"  ⚠️  SKIP: {filepath} not found")
        patches_skipped += 1
        return False

    content = filepath.read_text()

    if old_text not in content:
        # Check if already patched
        if new_text in content:
            print(f"  ✅ ALREADY APPLIED: {description}")
            patches_skipped += 1
            return True
        print(f"  ⚠️  SKIP: Could not find target text for: {description}")
        print(f"       File: {filepath}")
        print(f"       First 80 chars of search: {old_text[:80]}...")
        patches_skipped += 1
        return False

    if dry_run:
        print(f"  🔍 WOULD APPLY: {description}")
        return True

    # Backup
    bak = filepath.with_suffix(filepath.suffix + ".bak")
    if not bak.exists():
        shutil.copy2(filepath, bak)
        print(f"  📦 Backup: {bak}")

    content = content.replace(old_text, new_text, 1)
    filepath.write_text(content)
    print(f"  ✅ APPLIED: {description}")
    patches_applied += 1
    return True


def revert_all():
    """Revert all files from .bak backups."""
    for name, filepath in FILES.items():
        bak = filepath.with_suffix(filepath.suffix + ".bak")
        if bak.exists():
            shutil.copy2(bak, filepath)
            print(f"  ↩️  Reverted: {filepath}")
        else:
            print(f"  ⚠️  No backup for: {filepath}")


# =============================================================================
# FIX 1: Tennis grading — sport-aware _grade_single + _grade_game_predictions
# =============================================================================

def fix_tennis_grading(dry_run=False):
    print("\n🔧 FIX 1: Tennis scoring/grading")
    f = FILES["scheduler"]

    # 1a. Update _grade_game_predictions to pass sport_code
    apply_patch(f,
        old_text='''async def _grade_game_predictions(db: AsyncSession, game_id, home_score: int, away_score: int) -> int:
    """
    Grade all predictions for a single game that already has scores.
    Returns count of predictions graded.
    """
    preds = await db.execute(text("""
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
    for pred in preds.fetchall():
        result_val = _grade_single(
            pred.bet_type, pred.predicted_side,
            pred.line_at_prediction,
            home_score, away_score
        )''',
        new_text='''async def _grade_game_predictions(db: AsyncSession, game_id, home_score: int, away_score: int, sport_code: str = "") -> int:
    """
    Grade all predictions for a single game that already has scores.
    Returns count of predictions graded.
    """
    preds = await db.execute(text("""
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
    for pred in preds.fetchall():
        result_val = _grade_single(
            pred.bet_type, pred.predicted_side,
            pred.line_at_prediction,
            home_score, away_score,
            sport_code=sport_code
        )''',
        description="Add sport_code parameter to _grade_game_predictions",
        dry_run=dry_run,
    )

    # 1b. Update Phase 1 caller to pass sport_code
    apply_patch(f,
        old_text='''            logger.info(f"    Grading: {game.sport} {game.home_team_name} {game.home_score}-{game.away_score} {game.away_team_name}")
            cnt = await _grade_game_predictions(db, game.id, game.home_score, game.away_score)''',
        new_text='''            logger.info(f"    Grading: {game.sport} {game.home_team_name} {game.home_score}-{game.away_score} {game.away_team_name}")
            cnt = await _grade_game_predictions(db, game.id, game.home_score, game.away_score, sport_code=game.sport)''',
        description="Phase 1 grading: pass sport_code",
        dry_run=dry_run,
    )

    # 1c. Update Phase 2 caller to pass sport_code
    apply_patch(f,
        old_text='''                # Grade predictions using shared helper
                cnt = await _grade_game_predictions(db, game_uuid, grade_home, grade_away)''',
        new_text='''                # Grade predictions using shared helper
                cnt = await _grade_game_predictions(db, game_uuid, grade_home, grade_away, sport_code=sport_code)''',
        description="Phase 2 grading: pass sport_code",
        dry_run=dry_run,
    )

    # 1d. Update manual grading caller to pass sport_code
    apply_patch(f,
        old_text='''        # Grade predictions
        cnt = await _grade_game_predictions(db, row.id, home_score, away_score)''',
        new_text='''        # Grade predictions
        cnt = await _grade_game_predictions(db, row.id, home_score, away_score, sport_code=row.sport)''',
        description="Manual grading: pass sport_code",
        dry_run=dry_run,
    )

    # 1e. Replace _grade_single with sport-aware version
    apply_patch(f,
        old_text='''def _grade_single(
    bet_type: str, predicted_side: str, line: Optional[float],
    home_score: int, away_score: int,
) -> str:
    """Grade a single prediction. Returns 'win', 'loss', or 'push'."""
    score_diff = home_score - away_score  # Positive = home won
    total_points = home_score + away_score

    if bet_type == "spread":
        if line is None:
            return "void"
        # line is from the predicted side's perspective
        if predicted_side == "home":
            adjusted = score_diff + line  # home_score - away_score + home_spread
        else:  # away
            adjusted = -score_diff + line  # away effectively
        if adjusted > 0:
            return "win"
        elif adjusted < 0:
            return "loss"
        else:
            return "push"

    elif bet_type == "total":
        if line is None:
            return "void"
        if predicted_side == "over":
            if total_points > line:
                return "win"
            elif total_points < line:
                return "loss"
            else:
                return "push"
        else:  # under
            if total_points < line:
                return "win"
            elif total_points > line:
                return "loss"
            else:
                return "push"

    elif bet_type == "moneyline":
        if predicted_side == "home":
            return "win" if score_diff > 0 else ("loss" if score_diff < 0 else "push")
        else:
            return "win" if score_diff < 0 else ("loss" if score_diff > 0 else "push")

    return "void"''',
        new_text='''def _grade_single(
    bet_type: str, predicted_side: str, line: Optional[float],
    home_score: int, away_score: int,
    sport_code: str = "",
) -> str:
    """
    Grade a single prediction. Returns 'win', 'loss', 'push', or 'void'.

    Tennis fix: Odds API returns SETS won (e.g., 2-1), but spread/total lines
    are in GAMES (e.g., +2.5 games, 22.5 total games).  ESPN returns total
    games won.  We detect the format by checking total:
      - total <= 7 → SETS  → only moneyline is gradable
      - total > 7  → GAMES → spread/total are gradable, moneyline also works
                              (higher total games = winner in most cases)
    """
    score_diff = home_score - away_score  # Positive = home won
    total_points = home_score + away_score

    is_tennis = sport_code.upper() in ("ATP", "WTA")
    scores_are_sets = is_tennis and total_points <= 7

    if bet_type == "spread":
        if line is None:
            return "void"
        # Tennis: cannot grade spread with set-based scores
        if scores_are_sets:
            logger.debug(f"    Tennis spread voided: scores {home_score}-{away_score} are sets, not games")
            return "void"
        if predicted_side == "home":
            adjusted = score_diff + line
        else:
            adjusted = -score_diff + line
        if adjusted > 0:
            return "win"
        elif adjusted < 0:
            return "loss"
        else:
            return "push"

    elif bet_type == "total":
        if line is None:
            return "void"
        # Tennis: cannot grade total with set-based scores
        if scores_are_sets:
            logger.debug(f"    Tennis total voided: scores {home_score}-{away_score} are sets, not games")
            return "void"
        if predicted_side == "over":
            if total_points > line:
                return "win"
            elif total_points < line:
                return "loss"
            else:
                return "push"
        else:
            if total_points < line:
                return "win"
            elif total_points > line:
                return "loss"
            else:
                return "push"

    elif bet_type == "moneyline":
        # Moneyline works with BOTH sets and games:
        # higher score = winner (sets: 2>1, games: 15>12)
        if predicted_side == "home":
            return "win" if score_diff > 0 else ("loss" if score_diff < 0 else "push")
        else:
            return "win" if score_diff < 0 else ("loss" if score_diff > 0 else "push")

    return "void"''',
        description="Sport-aware _grade_single (tennis sets vs games detection)",
        dry_run=dry_run,
    )

    # 1f. Fix resolve_tennis_keys to return ALL active tournaments per tour
    apply_patch(f,
        old_text='''async def resolve_tennis_keys(api_key: str) -> dict:
    """Discover active ATP/WTA tournament keys from Odds API."""
    tennis_keys = {}
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                "https://api.the-odds-api.com/v4/sports",
                params={"apiKey": api_key},
            )
            if resp.status_code == 200:
                for s in resp.json():
                    key = s.get("key", "")
                    if key.startswith("tennis_atp_") and s.get("active"):
                        tennis_keys["ATP"] = key
                    elif key.startswith("tennis_wta_") and s.get("active"):
                        tennis_keys["WTA"] = key
    except Exception as e:
        logger.warning(f"  Failed to discover tennis keys: {e}")
    return tennis_keys''',
        new_text='''async def resolve_tennis_keys(api_key: str) -> dict:
    """
    Discover ALL active ATP/WTA tournament keys from Odds API.
    Returns dict like {"ATP": ["tennis_atp_miami_open", ...], "WTA": [...]}.
    Tours often have multiple simultaneous tournaments.
    """
    tennis_keys: dict = {"ATP": [], "WTA": []}
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                "https://api.the-odds-api.com/v4/sports",
                params={"apiKey": api_key},
            )
            if resp.status_code == 200:
                for s in resp.json():
                    key = s.get("key", "")
                    if key.startswith("tennis_atp_") and s.get("active"):
                        tennis_keys["ATP"].append(key)
                    elif key.startswith("tennis_wta_") and s.get("active"):
                        tennis_keys["WTA"].append(key)
    except Exception as e:
        logger.warning(f"  Failed to discover tennis keys: {e}")
    # Remove empty lists
    return {k: v for k, v in tennis_keys.items() if v}''',
        description="resolve_tennis_keys: return ALL active tournaments (not just one)",
        dry_run=dry_run,
    )

    # 1g. Update tennis key CONSUMERS to handle list format
    # In grade_predictions() Phase 2
    apply_patch(f,
        old_text='''            tennis_keys_to_try = []
            if sport_code in ("ATP", "WTA"):
                # 1) Currently active tournament
                if sport_code in tennis_keys:
                    tennis_keys_to_try.append(tennis_keys[sport_code])
                # 2) DB-stored key (tournament that was active when games were fetched)
                if db_api_key and db_api_key not in tennis_keys_to_try:
                    tennis_keys_to_try.append(db_api_key)
                if not tennis_keys_to_try:
                    logger.warning(f"    {sport_code}: No tennis tournament keys found")
                    continue
                api_sport_keys = tennis_keys_to_try''',
        new_text='''            tennis_keys_to_try = []
            if sport_code in ("ATP", "WTA"):
                # 1) ALL currently active tournaments (may be multiple)
                if sport_code in tennis_keys:
                    active = tennis_keys[sport_code]
                    if isinstance(active, list):
                        tennis_keys_to_try.extend(active)
                    else:
                        tennis_keys_to_try.append(active)
                # 2) DB-stored key (tournament that was active when games were fetched)
                if db_api_key and db_api_key not in tennis_keys_to_try:
                    tennis_keys_to_try.append(db_api_key)
                if not tennis_keys_to_try:
                    logger.warning(f"    {sport_code}: No tennis tournament keys found")
                    continue
                api_sport_keys = tennis_keys_to_try''',
        description="Grade predictions: handle list of tennis tournament keys",
        dry_run=dry_run,
    )

    # In refresh_odds()
    apply_patch(f,
        old_text='''            # Tennis needs tournament-specific key
            if sport_code in tennis_keys:
                api_sport_key = tennis_keys[sport_code]
            else:''',
        new_text='''            # Tennis needs tournament-specific key
            if sport_code in tennis_keys:
                active = tennis_keys[sport_code]
                api_sport_key = active[0] if isinstance(active, list) else active
            else:''',
        description="Refresh odds: handle list of tennis tournament keys",
        dry_run=dry_run,
    )


# =============================================================================
# FIX 2: Remove market-implied fallback (stop generating zero-edge predictions)
# =============================================================================

def fix_market_fallback(dry_run=False):
    print("\n🔧 FIX 2: Remove market-implied fallback (zero-edge) predictions")
    f = FILES["fetch_games"]

    # Replace the 3 fallback blocks (spread, total, moneyline) with skip logic
    apply_patch(f,
        old_text='''                else:
                    # Fallback: market-implied
                    if mkt_home >= mkt_away:
                        predictions_to_make.append(("home", mkt_home, line, home_price, 0.0))
                    else:
                        predictions_to_make.append(("away", mkt_away, -line if line else None, away_price, 0.0))
                    
        elif bet_type == "total":''',
        new_text='''                else:
                    # No ML model — skip (market-implied bets have zero edge
                    # and are mathematically guaranteed to lose after vig)
                    logger.debug(f"    Skipping {bet_type} spread: no ML model, market-implied only")
                    
        elif bet_type == "total":''',
        description="Remove market-implied SPREAD fallback",
        dry_run=dry_run,
    )

    apply_patch(f,
        old_text='''                else:
                    if mkt_over >= mkt_under:
                        predictions_to_make.append(("over", mkt_over, total, over_price, 0.0))
                    else:
                        predictions_to_make.append(("under", mkt_under, total, under_price, 0.0))
                    
        elif bet_type == "moneyline":''',
        new_text='''                else:
                    # No ML model — skip (market-implied bets have zero edge)
                    logger.debug(f"    Skipping {bet_type} total: no ML model, market-implied only")
                    
        elif bet_type == "moneyline":''',
        description="Remove market-implied TOTAL fallback",
        dry_run=dry_run,
    )

    apply_patch(f,
        old_text='''                else:
                    if mkt_home_fair >= mkt_away_fair:
                        predictions_to_make.append(("home", mkt_home_fair, None, home_ml, 0.0))
                    else:
                        predictions_to_make.append(("away", mkt_away_fair, None, away_ml, 0.0))''',
        new_text='''                else:
                    # No ML model — skip (market-implied bets have zero edge)
                    logger.debug(f"    Skipping {bet_type} moneyline: no ML model, market-implied only")''',
        description="Remove market-implied MONEYLINE fallback",
        dry_run=dry_run,
    )


# =============================================================================
# FIX 3: Fix probability clamping (too aggressive, destroying signal)
# =============================================================================

def fix_clamping(dry_run=False):
    print("\n🔧 FIX 3: Fix aggressive probability clamping in model_loader")
    f = FILES["model_loader"]

    # 3a. Replace _clamp_framework_output — much less aggressive
    apply_patch(f,
        old_text='''def _clamp_framework_output(prob: float) -> float:
    """
    Compress an individual framework's probability output to a realistic range.

    Sports betting reality: even the best single models achieve 55-60% accuracy.
    Any framework claiming > 65% is almost certainly using leaked features.

    Strategy:
    - [0.40, 0.60] → pass through (this range is well-calibrated per results)
    - (0.60, 0.65] → mild compression toward 0.60
    - (0.65, 1.00] → hard compression (broken model output)
    - Mirror for < 0.40
    """
    if 0.40 <= prob <= 0.60:
        return prob

    if prob > 0.60:
        # Above 0.60: compress excess with diminishing returns
        # 0.65 → 0.62, 0.70 → 0.63, 0.80 → 0.635, 0.95 → 0.645
        excess = prob - 0.60
        compressed = 0.60 + excess * 0.10  # 10% of excess preserved
        return min(compressed, 0.65)

    if prob < 0.40:
        # Mirror: below 0.40 compress toward 0.40
        deficit = 0.40 - prob
        compressed = 0.40 - deficit * 0.10
        return max(compressed, 0.35)

    return prob''',
        new_text='''def _clamp_framework_output(prob: float) -> float:
    """
    Soft-clamp an individual framework's probability output.

    Only compress truly extreme outputs (> 0.80 or < 0.20) that indicate
    data leakage or broken models.  Let the calibrator handle normal range
    adjustments instead of destroying signal here.

    Strategy:
    - [0.25, 0.75] → pass through (let calibrator handle fine-tuning)
    - (0.75, 0.90] → mild compression
    - (0.90, 1.00] → hard compression (likely data leakage)
    - Mirror for low end
    """
    if 0.25 <= prob <= 0.75:
        return prob

    if prob > 0.75:
        # Above 0.75: compress excess
        # 0.80 → 0.775, 0.85 → 0.80, 0.90 → 0.825, 0.95 → 0.85
        excess = prob - 0.75
        compressed = 0.75 + excess * 0.50  # 50% of excess preserved
        return min(compressed, 0.85)

    if prob < 0.25:
        # Mirror
        deficit = 0.25 - prob
        compressed = 0.25 - deficit * 0.50
        return max(compressed, 0.15)

    return prob''',
        description="Less aggressive _clamp_framework_output (was destroying 90% of signal)",
        dry_run=dry_run,
    )

    # 3b. Remove extreme weight penalties (they destroy ensemble signal)
    apply_patch(f,
        old_text='''        # Penalize weight of frameworks producing extreme outputs
        # A framework outputting 0.95 is almost certainly data-leaked
        # Reduce its ensemble influence so honest frameworks dominate
        if raw > 0.70 or raw < 0.30:
            weight_penalties[fw] = 0.10  # 90% weight reduction
            logger.info(f"    ⚠️  {fw} weight penalized 90% (raw={raw:.3f}, likely data leakage)")
        elif raw > 0.65 or raw < 0.35:
            weight_penalties[fw] = 0.30  # 70% weight reduction
        else:
            weight_penalties[fw] = 1.00  # No penalty''',
        new_text='''        # Penalize weight of frameworks producing extreme outputs
        # Only penalize truly absurd outputs (> 0.90) that indicate broken models
        if raw > 0.90 or raw < 0.10:
            weight_penalties[fw] = 0.20  # 80% weight reduction (likely broken)
            logger.info(f"    ⚠️  {fw} weight penalized 80% (raw={raw:.3f}, likely data leakage)")
        elif raw > 0.80 or raw < 0.20:
            weight_penalties[fw] = 0.50  # 50% weight reduction
        else:
            weight_penalties[fw] = 1.00  # No penalty''',
        description="Softer weight penalties (was penalizing normal signals)",
        dry_run=dry_run,
    )

    # 3c. Widen static fallback shrinkage caps
    apply_patch(f,
        old_text='''        # Fallback: static shrinkage toward 50% (used before enough data to train)
        SHRINKAGE_FACTOR = 0.667
        MAX_PROBABILITY = 0.62
        MIN_PROBABILITY = 0.38''',
        new_text='''        # Fallback: static shrinkage toward 50% (used before enough data to train)
        SHRINKAGE_FACTOR = 0.80
        MAX_PROBABILITY = 0.70
        MIN_PROBABILITY = 0.30''',
        description="Widen static fallback shrinkage (was capping at 0.62)",
        dry_run=dry_run,
    )


# =============================================================================
# FIX 4: Widen calibrator hard caps
# =============================================================================

def fix_calibrator(dry_run=False):
    print("\n🔧 FIX 4: Widen calibrator hard caps")
    f = FILES["calibrator"]

    apply_patch(f,
        old_text='''        # Hard caps: no sports prediction should exceed these bounds
        return float(np.clip(calibrated, 0.38, 0.62))''',
        new_text='''        # Hard caps: allow wider range to preserve genuine signal
        # The calibrator itself should handle shrinkage; these are safety bounds only
        return float(np.clip(calibrated, 0.30, 0.70))''',
        description="Widen calibrator hard caps from [0.38, 0.62] to [0.30, 0.70]",
        dry_run=dry_run,
    )


# =============================================================================
# FIX 5: Assign tiers AFTER calibration (currently uses pre-calibrated prob)
# =============================================================================

def fix_tier_assignment(dry_run=False):
    print("\n🔧 FIX 5: Tier assignment is already post-calibration (verify)")
    # In fetch_games.py, predict_probability() already returns calibrated values.
    # The tier assignment at line 677 uses the returned probability which IS calibrated.
    # So this is actually correct in the current code. The problem was the
    # calibration itself squishing everything, which we fixed in Fix 3.
    print("  ℹ️  No change needed — tier assignment already uses calibrated probability.")
    print("     The issue was the calibration squishing (fixed in Fix 3).")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="ROYALEY Bug Fix Patcher")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without applying")
    parser.add_argument("--revert", action="store_true", help="Revert all changes from backups")
    args = parser.parse_args()

    if args.revert:
        print("↩️  Reverting all changes...")
        revert_all()
        print("Done!")
        return

    mode = "DRY RUN" if args.dry_run else "APPLYING"
    print("=" * 60)
    print(f"ROYALEY Bug Fix Patcher — {mode}")
    print("=" * 60)

    # Verify files exist
    print("\n📁 Checking files...")
    all_found = True
    for name, path in FILES.items():
        if path.exists():
            print(f"  ✅ {name}: {path}")
        else:
            print(f"  ❌ {name}: {path} NOT FOUND")
            all_found = False

    if not all_found:
        print("\n❌ Some files not found. Are you in the right directory?")
        print(f"   Expected project root: {PROJECT_ROOT}")
        print("   Run this from your project root: cd /nvme0n1-disk/royaley && python3 apply_fixes.py")
        sys.exit(1)

    # Apply fixes
    fix_tennis_grading(args.dry_run)
    fix_market_fallback(args.dry_run)
    fix_clamping(args.dry_run)
    fix_calibrator(args.dry_run)
    fix_tier_assignment(args.dry_run)

    print("\n" + "=" * 60)
    if args.dry_run:
        print(f"DRY RUN complete. Would apply {patches_applied + patches_skipped} patches.")
        print("Run without --dry-run to apply changes.")
    else:
        print(f"✅ Applied {patches_applied} patches, skipped {patches_skipped}")
        if patches_applied > 0:
            print("\n📋 Next steps:")
            print("   1. Rebuild the container:")
            print("      docker compose up -d --build api")
            print("")
            print("   2. Run diagnostics to verify:")
            print("      bash diagnose.sh")
            print("")
            print("   3. Retrain the calibrator (once you have ~50+ new graded predictions):")
            print("      docker exec royaley_api python -m app.pipeline.calibrator --train")
            print("")
            print("   4. To revert all changes:")
            print("      python3 apply_fixes.py --revert")
    print("=" * 60)


if __name__ == "__main__":
    main()