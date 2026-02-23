#!/usr/bin/env python3
"""
ROYALEY - Bug Fix Patcher v2 (CORRECTED)
==========================================
Based on ACTUAL diagnostic data:

  ML Model (edge > 0):  43.5% win rate  -> -$6,189  <- INVERTED SIGNAL
  Market Fallback:       56.1% win rate  -> +$569    <- PROFITABLE
  Tier B (highest conf): 44.6% actual    <- WORST tier
  Tier D (lowest conf):  51.4% actual    <- BEST tier

v2 Strategy:
  FIX 1: Tennis grading (sets vs games scoring mismatch)
  FIX 2: Tennis multi-tournament support
  FIX 3: ML model inversion (flip predictions)
  FIX 4: Softer probability clamping
  FIX 5: Wider calibrator caps
  KEEP market fallback (it's profitable!)

Usage:
    cd /nvme0n1-disk/royaley
    python3 apply_fixes_v2.py --dry-run
    python3 apply_fixes_v2.py
    python3 apply_fixes_v2.py --revert
"""

import re
import sys
import shutil
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
FILES = {
    "scheduler":    PROJECT_ROOT / "app" / "pipeline" / "scheduler.py",
    "model_loader": PROJECT_ROOT / "app" / "pipeline" / "model_loader.py",
    "fetch_games":  PROJECT_ROOT / "app" / "pipeline" / "fetch_games.py",
    "calibrator":   PROJECT_ROOT / "app" / "pipeline" / "calibrator.py",
}

applied = 0
skipped = 0


def backup_file(filepath):
    bak = filepath.with_suffix(filepath.suffix + ".bak2")
    if not bak.exists():
        shutil.copy2(filepath, bak)
        print(f"  📦 Backup: {bak}")


def patch(filepath, old, new, desc, dry_run=False):
    global applied, skipped
    if not filepath.exists():
        print(f"  ⚠️  SKIP (file missing): {desc}")
        skipped += 1
        return False
    content = filepath.read_text()
    if old not in content:
        if new in content:
            print(f"  ✅ ALREADY APPLIED: {desc}")
            skipped += 1
            return True
        print(f"  ⚠️  SKIP (text not found): {desc}")
        print(f"       First 80 chars: {old[:80]}...")
        skipped += 1
        return False
    if dry_run:
        print(f"  🔍 WOULD APPLY: {desc}")
        return True
    backup_file(filepath)
    content = content.replace(old, new, 1)
    filepath.write_text(content)
    print(f"  ✅ APPLIED: {desc}")
    applied += 1
    return True


def regex_patch(filepath, pattern, replacement, desc, check=None, dry_run=False):
    global applied, skipped
    if not filepath.exists():
        print(f"  ⚠️  SKIP (file missing): {desc}")
        skipped += 1
        return False
    content = filepath.read_text()
    if check and check in content:
        print(f"  ✅ ALREADY APPLIED: {desc}")
        skipped += 1
        return True
    if not re.search(pattern, content):
        print(f"  ⚠️  SKIP (pattern not found): {desc}")
        skipped += 1
        return False
    if dry_run:
        print(f"  🔍 WOULD APPLY: {desc}")
        return True
    backup_file(filepath)
    content = re.sub(pattern, replacement, content, count=1)
    filepath.write_text(content)
    print(f"  ✅ APPLIED: {desc}")
    applied += 1
    return True


def revert_all():
    for name, fp in FILES.items():
        bak2 = fp.with_suffix(fp.suffix + ".bak2")
        bak1 = fp.with_suffix(fp.suffix + ".bak")
        if bak2.exists():
            shutil.copy2(bak2, fp)
            print(f"  ↩️  Reverted: {fp}")
        elif bak1.exists():
            shutil.copy2(bak1, fp)
            print(f"  ↩️  Reverted from .bak: {fp}")
        else:
            print(f"  ⚠️  No backup for: {fp}")


# =============================================================================
# FIX 1: Tennis grading
# =============================================================================

def fix_tennis_grading(dry_run=False):
    print("\n🔧 FIX 1: Tennis scoring/grading")
    f = FILES["scheduler"]

    # 1a. Add sport_code to _grade_game_predictions (REGEX for whitespace flexibility)
    regex_patch(f,
        pattern=r'(async def _grade_game_predictions\(db:\s*AsyncSession,\s*game_id,\s*home_score:\s*int,\s*away_score:\s*int)\)\s*->\s*int:',
        replacement=r'\g<1>, sport_code: str = "") -> int:',
        desc="Add sport_code param to _grade_game_predictions",
        check='sport_code: str = "") -> int:',
        dry_run=dry_run,
    )

    # 1b. Pass sport_code in _grade_single call (REGEX)
    regex_patch(f,
        pattern=r'(result_val\s*=\s*_grade_single\(\s*\n\s*pred\.bet_type,\s*pred\.predicted_side,\s*\n\s*pred\.line_at_prediction,\s*\n\s*home_score,\s*away_score)\s*\n\s*\)',
        replacement=r"""\g<1>,
            sport_code=sport_code
        )""",
        desc="Pass sport_code to _grade_single call",
        check='sport_code=sport_code',
        dry_run=dry_run,
    )

    # 1c. Phase 1 caller (REGEX)
    regex_patch(f,
        pattern=r'(cnt\s*=\s*await\s+_grade_game_predictions\(db,\s*game\.id,\s*game\.home_score,\s*game\.away_score)\)',
        replacement=r'\g<1>, sport_code=game.sport)',
        desc="Phase 1 grading: pass sport_code",
        check='game.away_score, sport_code=game.sport)',
        dry_run=dry_run,
    )

    # 1d. Phase 2 caller
    patch(f,
        old='                cnt = await _grade_game_predictions(db, game_uuid, grade_home, grade_away)',
        new='                cnt = await _grade_game_predictions(db, game_uuid, grade_home, grade_away, sport_code=sport_code)',
        desc="Phase 2 grading: pass sport_code",
        dry_run=dry_run,
    )

    # 1e. Manual grading caller
    patch(f,
        old='        cnt = await _grade_game_predictions(db, row.id, home_score, away_score)',
        new='        cnt = await _grade_game_predictions(db, row.id, home_score, away_score, sport_code=row.sport)',
        desc="Manual grading: pass sport_code",
        dry_run=dry_run,
    )

    # 1f. Replace _grade_single with sport-aware version
    patch(f,
        old='''def _grade_single(
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
        new='''def _grade_single(
    bet_type: str, predicted_side: str, line: Optional[float],
    home_score: int, away_score: int,
    sport_code: str = "",
) -> str:
    """
    Grade a single prediction. Returns 'win', 'loss', 'push', or 'void'.
    Tennis fix: Odds API returns SETS (e.g. 2-1) but lines are in GAMES.
    Detect by total: <= 7 = sets, > 7 = games.
    """
    score_diff = home_score - away_score
    total_points = home_score + away_score
    is_tennis = sport_code.upper() in ("ATP", "WTA")
    scores_are_sets = is_tennis and total_points <= 7

    if bet_type == "spread":
        if line is None:
            return "void"
        if scores_are_sets:
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
        if scores_are_sets:
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
        if predicted_side == "home":
            return "win" if score_diff > 0 else ("loss" if score_diff < 0 else "push")
        else:
            return "win" if score_diff < 0 else ("loss" if score_diff > 0 else "push")

    return "void"''',
        desc="Sport-aware _grade_single (tennis sets vs games)",
        dry_run=dry_run,
    )


# =============================================================================
# FIX 2: Tennis multi-tournament
# =============================================================================

def fix_tennis_tournaments(dry_run=False):
    print("\n🔧 FIX 2: Tennis multi-tournament support")
    f = FILES["scheduler"]

    patch(f,
        old='''async def resolve_tennis_keys(api_key: str) -> dict:
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
        new='''async def resolve_tennis_keys(api_key: str) -> dict:
    """
    Discover ALL active ATP/WTA tournament keys from Odds API.
    Returns {"ATP": ["tennis_atp_miami_open", ...], "WTA": [...]}.
    Tours often run simultaneous events.
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
    return {k: v for k, v in tennis_keys.items() if v}''',
        desc="resolve_tennis_keys: return ALL active tournaments",
        dry_run=dry_run,
    )

    patch(f,
        old='''            tennis_keys_to_try = []
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
        new='''            tennis_keys_to_try = []
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
        desc="Grade predictions: handle list of tennis keys",
        dry_run=dry_run,
    )

    patch(f,
        old='''            # Tennis needs tournament-specific key
            if sport_code in tennis_keys:
                api_sport_key = tennis_keys[sport_code]
            else:''',
        new='''            # Tennis needs tournament-specific key
            if sport_code in tennis_keys:
                active = tennis_keys[sport_code]
                api_sport_key = active[0] if isinstance(active, list) else active
            else:''',
        desc="Refresh odds: handle list of tennis keys",
        dry_run=dry_run,
    )


# =============================================================================
# FIX 3: ML Model Inversion (THE KEY FIX)
# =============================================================================

def fix_ml_inversion(dry_run=False):
    print("\n🔧 FIX 3: ML model inversion (43.5% → flip to ~56.5%)")
    f = FILES["fetch_games"]

    # 3a. Invert SPREAD
    patch(f,
        old='''                if model_prob:
                    # ML model: p1 = P(home covers)
                    home_prob = model_prob[0]
                    away_prob = model_prob[1]
                    market_prob_for_edge = mkt_home  # Compare model vs market
                    if home_prob >= away_prob:
                        edge = home_prob - mkt_home
                        predictions_to_make.append(("home", home_prob, line, home_price, edge))
                    else:
                        edge = away_prob - mkt_away
                        predictions_to_make.append(("away", away_prob, -line if line else None, away_price, edge))''',
        new='''                if model_prob:
                    # ML model: p1 = P(home covers) — INVERTED per diagnostics
                    # Model at 43.5% win rate = backwards signal. Flip picks.
                    home_prob = model_prob[1]  # SWAP: was [0]
                    away_prob = model_prob[0]  # SWAP: was [1]
                    market_prob_for_edge = mkt_home
                    if home_prob >= away_prob:
                        edge = home_prob - mkt_home
                        predictions_to_make.append(("home", home_prob, line, home_price, edge))
                    else:
                        edge = away_prob - mkt_away
                        predictions_to_make.append(("away", away_prob, -line if line else None, away_price, edge))''',
        desc="Invert ML SPREAD predictions (swap [0]/[1])",
        dry_run=dry_run,
    )

    # 3b. Invert TOTAL
    patch(f,
        old='''                if model_prob:
                    # ML model: p1 = P(over)
                    over_prob = model_prob[0]
                    under_prob = model_prob[1]
                    if over_prob >= under_prob:
                        edge = over_prob - mkt_over
                        predictions_to_make.append(("over", over_prob, total, over_price, edge))
                    else:
                        edge = under_prob - mkt_under
                        predictions_to_make.append(("under", under_prob, total, under_price, edge))''',
        new='''                if model_prob:
                    # ML model: p1 = P(over) — INVERTED per diagnostics
                    over_prob = model_prob[1]   # SWAP: was [0]
                    under_prob = model_prob[0]  # SWAP: was [1]
                    if over_prob >= under_prob:
                        edge = over_prob - mkt_over
                        predictions_to_make.append(("over", over_prob, total, over_price, edge))
                    else:
                        edge = under_prob - mkt_under
                        predictions_to_make.append(("under", under_prob, total, under_price, edge))''',
        desc="Invert ML TOTAL predictions (swap [0]/[1])",
        dry_run=dry_run,
    )

    # 3c. Invert MONEYLINE
    patch(f,
        old='''                if model_prob:
                    # ML model: p1 = P(home wins)
                    home_prob = model_prob[0]
                    away_prob = model_prob[1]
                    if home_prob >= away_prob:
                        edge = home_prob - mkt_home_fair
                        predictions_to_make.append(("home", home_prob, None, home_ml, edge))
                    else:
                        edge = away_prob - mkt_away_fair
                        predictions_to_make.append(("away", away_prob, None, away_ml, edge))''',
        new='''                if model_prob:
                    # ML model: p1 = P(home wins) — INVERTED per diagnostics
                    home_prob = model_prob[1]  # SWAP: was [0]
                    away_prob = model_prob[0]  # SWAP: was [1]
                    if home_prob >= away_prob:
                        edge = home_prob - mkt_home_fair
                        predictions_to_make.append(("home", home_prob, None, home_ml, edge))
                    else:
                        edge = away_prob - mkt_away_fair
                        predictions_to_make.append(("away", away_prob, None, away_ml, edge))''',
        desc="Invert ML MONEYLINE predictions (swap [0]/[1])",
        dry_run=dry_run,
    )


# =============================================================================
# FIX 4: Softer probability clamping
# =============================================================================

def fix_clamping(dry_run=False):
    print("\n🔧 FIX 4: Softer probability clamping")
    f = FILES["model_loader"]

    # 4a. Replace _clamp_framework_output
    patch(f,
        old='''def _clamp_framework_output(prob: float) -> float:
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
        new='''def _clamp_framework_output(prob: float) -> float:
    """
    Soft-clamp framework output. Only compress extreme outputs (> 0.80)
    that indicate data leakage. Let calibrator handle normal adjustments.
    """
    if 0.25 <= prob <= 0.75:
        return prob

    if prob > 0.75:
        excess = prob - 0.75
        compressed = 0.75 + excess * 0.50
        return min(compressed, 0.85)

    if prob < 0.25:
        deficit = 0.25 - prob
        compressed = 0.25 - deficit * 0.50
        return max(compressed, 0.15)

    return prob''',
        desc="Softer _clamp_framework_output [0.25-0.75] passthrough",
        dry_run=dry_run,
    )

    # 4b. Softer weight penalties
    patch(f,
        old='''        # Penalize weight of frameworks producing extreme outputs
        # A framework outputting 0.95 is almost certainly data-leaked
        # Reduce its ensemble influence so honest frameworks dominate
        if raw > 0.70 or raw < 0.30:
            weight_penalties[fw] = 0.10  # 90% weight reduction
            logger.info(f"    ⚠️  {fw} weight penalized 90% (raw={raw:.3f}, likely data leakage)")
        elif raw > 0.65 or raw < 0.35:
            weight_penalties[fw] = 0.30  # 70% weight reduction
        else:
            weight_penalties[fw] = 1.00  # No penalty''',
        new='''        # Only penalize truly absurd outputs (> 0.90) indicating broken models
        if raw > 0.90 or raw < 0.10:
            weight_penalties[fw] = 0.20  # 80% weight reduction
            logger.info(f"    ⚠️  {fw} weight penalized 80% (raw={raw:.3f}, likely data leakage)")
        elif raw > 0.80 or raw < 0.20:
            weight_penalties[fw] = 0.50  # 50% weight reduction
        else:
            weight_penalties[fw] = 1.00  # No penalty''',
        desc="Softer weight penalties (only penalize > 0.90)",
        dry_run=dry_run,
    )

    # 4c. Wider static shrinkage
    patch(f,
        old='''        # Fallback: static shrinkage toward 50% (used before enough data to train)
        SHRINKAGE_FACTOR = 0.667
        MAX_PROBABILITY = 0.62
        MIN_PROBABILITY = 0.38''',
        new='''        # Fallback: static shrinkage toward 50% (used before enough data to train)
        SHRINKAGE_FACTOR = 0.80
        MAX_PROBABILITY = 0.70
        MIN_PROBABILITY = 0.30''',
        desc="Wider static shrinkage [0.38-0.62] → [0.30-0.70]",
        dry_run=dry_run,
    )


# =============================================================================
# FIX 5: Wider calibrator hard caps
# =============================================================================

def fix_calibrator(dry_run=False):
    print("\n🔧 FIX 5: Wider calibrator hard caps")
    f = FILES["calibrator"]

    patch(f,
        old='''        # Hard caps: no sports prediction should exceed these bounds
        return float(np.clip(calibrated, 0.38, 0.62))''',
        new='''        # Hard caps: wider range to let genuine signal through
        return float(np.clip(calibrated, 0.30, 0.70))''',
        desc="Widen calibrator caps [0.38,0.62] → [0.30,0.70]",
        dry_run=dry_run,
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="ROYALEY Bug Fix Patcher v2")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--revert", action="store_true")
    args = parser.parse_args()

    if args.revert:
        print("↩️  Reverting v2 changes...")
        revert_all()
        print("Done!")
        return

    mode = "DRY RUN" if args.dry_run else "APPLYING"
    print("=" * 60)
    print(f"ROYALEY Bug Fix Patcher v2 — {mode}")
    print("=" * 60)
    print()
    print("📊 Based on diagnostic data:")
    print("   ML Model:  43.5% win rate → -$6,189 (INVERTED)")
    print("   Fallback:  56.1% win rate → +$569   (KEEP)")
    print("   Tier B:    44.6% actual   (worst = highest conf)")
    print("   Tier D:    51.4% actual   (best  = lowest conf)")
    print()
    print("🎯 Strategy: Flip ML picks + fix tennis + soften clamping")
    print()

    missing = False
    for name, path in FILES.items():
        exists = path.exists()
        print(f"  {'✅' if exists else '❌'} {name}: {path}")
        if not exists:
            missing = True
    if missing:
        print(f"\n❌ Run from project root: cd /nvme0n1-disk/royaley")
        sys.exit(1)

    fix_tennis_grading(args.dry_run)
    fix_tennis_tournaments(args.dry_run)
    fix_ml_inversion(args.dry_run)
    fix_clamping(args.dry_run)
    fix_calibrator(args.dry_run)

    print()
    print("=" * 60)
    if args.dry_run:
        print("DRY RUN complete. Run without --dry-run to apply.")
    else:
        print(f"✅ Applied {applied} patches, skipped {skipped}")
        if applied > 0:
            print("""
📋 Next steps:
   1. Rebuild:  docker compose up -d --build api
   2. Monitor:  docker logs -f royaley_api 2>&1 | grep -E "predict|🎯"
   3. After ~50 new graded predictions, check improvement:
      bash diagnose.sh
   4. Retrain calibrator once inversion is confirmed:
      docker exec royaley_api python -m app.pipeline.calibrator --train
   5. Revert:   python3 apply_fixes_v2.py --revert

⚠️  Market fallback KEPT (profitable at 56.1%).
    ML inversion should bring ML from 43.5% → ~56.5%.
    Monitor 2-3 days to confirm.""")
    print("=" * 60)


if __name__ == "__main__":
    main()