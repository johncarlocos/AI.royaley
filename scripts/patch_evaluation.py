#!/usr/bin/env python3
"""
ROYALEY V5.3 PATCH — FIX DATA LOADING FOR EVALUATION
=====================================================
Applies ON TOP of v5.2 patch.

ROOT CAUSES (from evaluation scorecard analysis):
  1. CSV loading concatenates ALL files in sport dir (main + situation + team + weather)
     vertically, creating column misalignment and duplicate rows.
     FIX: Only load the MAIN csv file, skip sub-type files.

  2. Garbage/placeholder rows with NaT dates and zero scores are kept.
     For moneyline target (home_win), garbage rows have home_win=0 (not NaN),
     so dropna() never removes them. The 80/20 split puts all garbage at the end
     as "validation" → evaluation against all-zero targets → AUC=0.500.
     FIX: Parse dates, drop NaT rows, sort by date before splitting.

  3. Feature engineering runs AFTER the split but should run BEFORE feature
     column identification to ensure derived features are included.
     FIX: Already handled by v5.2 patch (inject before exclude_cols).

  4. FIX 7 diagnostic crash: references 'df' outside function scope.
     FIX: Remove the broken diagnostic injection.

RESULT: Validation set will contain real games with real targets and features.
"""

import re
import sys
import shutil
from pathlib import Path


# ============================================================================
# CONFIGURATION
# ============================================================================
EVALUATE_PY = Path("/app/scripts/evaluate_models.py")

# Try alternate locations
if not EVALUATE_PY.exists():
    alt_paths = [
        Path(__file__).parent / "evaluate_models.py",
        Path("/home/claude/royaley/AI.royaley/scripts/evaluate_models.py"),
    ]
    for p in alt_paths:
        if p.exists():
            EVALUATE_PY = p
            break


def main():
    print("=" * 70)
    print("ROYALEY V5.3 PATCH — FIX DATA LOADING")
    print("=" * 70)

    if not EVALUATE_PY.exists():
        print(f"\n❌ evaluate_models.py not found at {EVALUATE_PY}")
        print("Please update EVALUATE_PY path at top of this script.")
        sys.exit(1)

    print(f"\n📂 Target: {EVALUATE_PY}")

    # Backup
    backup = EVALUATE_PY.with_suffix('.py.bak_v53')
    shutil.copy2(EVALUATE_PY, backup)
    print(f"💾 Backup: {backup}")

    content = EVALUATE_PY.read_text()
    original_len = len(content)
    patches_applied = 0
    patches_failed = []

    # ========================================================================
    # FIX A: REPLACE CSV LOADING LOGIC
    # ========================================================================
    print("\n--- FIX A: Fix CSV loading (main file only + filter garbage rows) ---")

    # Find the load_validation_data function and replace the CSV loading + filtering
    # We need to replace from "for csv_path in csv_paths:" through "if len(df) < 30: return None"
    # (the first one, before feature column identification)

    # Strategy: Find and replace the entire body of load_validation_data
    # from the csv loading loop through the validation split

    OLD_LOADING_PATTERN = re.compile(
        r'(    for csv_path in csv_paths:\n)'       # Start of loading loop
        r'(.*?)'                                     # Everything in between
        r'(    # Identify feature columns)',          # End marker
        re.DOTALL
    )

    NEW_LOADING = '''    for csv_path in csv_paths:
        if csv_path is None or not csv_path.exists():
            continue

        sport_dir = csv_path / sport
        if sport_dir.exists() and sport_dir.is_dir():
            # === V5.3 FIX A: Load ONLY the main CSV, not sub-type files ===
            # Sub-files (situation, team, weather) have different column structures
            # and should NOT be concatenated vertically.
            files = sorted(sport_dir.glob("ml_features_*_20*.csv"))
            # Filter out sub-type files
            main_files = [f for f in files
                          if '_situation_' not in f.name
                          and '_team_' not in f.name
                          and '_weather_' not in f.name
                          and '_player_' not in f.name]
            if not main_files:
                main_files = files  # Fallback to all files if naming differs
            if main_files:
                try:
                    df = pd.read_csv(main_files[0])
                    logger.info(f"  Loaded {main_files[0].name}: {len(df)} rows x {len(df.columns)} cols")
                except Exception as e:
                    logger.warning(f"  Failed to load {main_files[0]}: {e}")
                    continue
                break
        else:
            # Try flat structure
            pattern = f"ml_features_{sport}_*.csv"
            files = sorted(csv_path.glob(pattern))
            main_files = [f for f in files
                          if '_situation_' not in f.name
                          and '_team_' not in f.name
                          and '_weather_' not in f.name
                          and '_player_' not in f.name]
            if not main_files:
                main_files = files
            if main_files:
                try:
                    df = pd.read_csv(main_files[0])
                    logger.info(f"  Loaded {main_files[0].name}: {len(df)} rows x {len(df.columns)} cols")
                except Exception as e:
                    logger.warning(f"  Failed to load {main_files[0]}: {e}")
                    continue
                break

    if df is None or len(df) < 30:
        return None

    # === V5.3 FIX A: Filter out garbage/placeholder rows ===
    # Many CSVs contain placeholder rows with NaT dates and zero scores.
    # These MUST be removed before the train/validation split.
    pre_filter = len(df)

    # Step 1: Parse dates and drop NaT
    if 'scheduled_at' in df.columns:
        df['_parsed_date'] = pd.to_datetime(df['scheduled_at'], errors='coerce')
        nat_count = df['_parsed_date'].isna().sum()
        if nat_count > 0:
            df = df[df['_parsed_date'].notna()].copy()
            logger.info(f"  v5.3: Dropped {nat_count} rows with NaT dates ({len(df)} remaining)")
    elif 'date' in df.columns:
        df['_parsed_date'] = pd.to_datetime(df['date'], errors='coerce')
        nat_count = df['_parsed_date'].isna().sum()
        if nat_count > 0:
            df = df[df['_parsed_date'].notna()].copy()
            logger.info(f"  v5.3: Dropped {nat_count} rows with NaT dates ({len(df)} remaining)")

    # Step 2: Drop rows where ALL scores are zero (unplayed games)
    score_cols = [c for c in ['home_score', 'away_score', 'score_margin'] if c in df.columns]
    if score_cols:
        all_zero = (df[score_cols] == 0).all(axis=1)
        # But don't drop rows that legitimately ended 0-0 — check if home_win is also 0
        # and there are no odds. Real games have some non-zero score or valid odds.
        if 'home_win' in df.columns:
            has_odds = False
            for oc in ['spread_close', 'total_close', 'moneyline_home_close']:
                if oc in df.columns:
                    has_odds = df[oc].notna()
                    break
            if has_odds is not False:
                garbage = all_zero & ~has_odds & (df['home_win'] == 0)
            else:
                garbage = all_zero
        else:
            garbage = all_zero
        garbage_count = garbage.sum()
        if garbage_count > 0:
            df = df[~garbage].copy()
            logger.info(f"  v5.3: Dropped {garbage_count} zero-score/no-odds rows ({len(df)} remaining)")

    # Step 3: Sort by date for proper chronological split
    if '_parsed_date' in df.columns:
        df = df.sort_values('_parsed_date').reset_index(drop=True)
        df = df.drop(columns=['_parsed_date'])

    logger.info(f"  v5.3: After filtering: {len(df)} rows (was {pre_filter})")

    if len(df) < 30:
        return None

    # Ensure target exists
    if target_col not in df.columns:
        # Try to reconstruct
        if 'home_score' in df.columns and 'away_score' in df.columns:
            margin = df['home_score'] - df['away_score']
            if target_col == 'home_win':
                df['home_win'] = (margin > 0).astype(int)
            elif target_col == 'spread_result':
                for col in ['spread_close', 'spread_line', 'home_spread', 'home_line']:
                    if col in df.columns:
                        df['spread_result'] = (margin > -df[col]).astype(float)
                        break
            elif target_col == 'over_result':
                total_pts = df['home_score'] + df['away_score']
                for col in ['total_close', 'total_line', 'over_under_line']:
                    if col in df.columns:
                        df['over_result'] = (total_pts > df[col]).astype(float)
                        break

    if target_col not in df.columns:
        return None

    # Drop rows with NaN target
    df = df.dropna(subset=[target_col])

    if len(df) < 30:
        return None

    # Identify feature columns'''

    match = OLD_LOADING_PATTERN.search(content)
    if match and 'V5.3 FIX A' not in content:
        content = content[:match.start()] + NEW_LOADING + content[match.end():]
        patches_applied += 1
        print(f"  ✅ [FIX {patches_applied}] Replaced CSV loading with filtered version")
    elif 'V5.3 FIX A' in content:
        print("  ⏭️  V5.3 data loading fix already applied")
    else:
        patches_failed.append("FIX A: Could not find CSV loading pattern")
        print("  ❌ Could not find CSV loading pattern in load_validation_data")
        # Show what we're looking for
        if 'for csv_path in csv_paths:' in content:
            print("     'for csv_path in csv_paths:' found, but regex didn't match")
        else:
            print("     'for csv_path in csv_paths:' NOT found in file")

    # ========================================================================
    # FIX B: Remove broken FIX 7 diagnostic (causes NameError crash)
    # ========================================================================
    print("\n--- FIX B: Remove broken v5.2 FIX 7 diagnostic ---")

    # The diagnostic references 'df' in a scope where it's not defined
    diag_pattern = re.compile(
        r'\n    # V5\.2 FIX 7: Pipeline diagnostics\n'
        r'    import logging as _logdiag\n'
        r'.*?'
        r'spread_value.*?\n'
        r'    \)\n',
        re.DOTALL
    )

    diag_match = diag_pattern.search(content)
    if diag_match:
        content = content[:diag_match.start()] + '\n' + content[diag_match.end():]
        patches_applied += 1
        print(f"  ✅ [FIX {patches_applied}] Removed broken v5.2 FIX 7 diagnostic")
    elif 'v5.2 PIPELINE' not in content:
        print("  ⏭️  No broken diagnostic found")
    else:
        # Try simpler pattern
        simple_diag = re.compile(
            r'    # V5\.2 FIX 7.*?(?=\n    return |\n    #|\ndef )',
            re.DOTALL
        )
        simple_match = simple_diag.search(content)
        if simple_match:
            content = content[:simple_match.start()] + content[simple_match.end():]
            patches_applied += 1
            print(f"  ✅ [FIX {patches_applied}] Removed broken diagnostic (simple match)")
        else:
            patches_failed.append("FIX B: Could not remove broken diagnostic")
            print("  ❌ Could not find diagnostic pattern to remove")

    # ========================================================================
    # FIX C: Ensure the exclude_cols set includes additional meta columns
    # ========================================================================
    print("\n--- FIX C: Expand exclude_cols to catch more meta columns ---")

    OLD_EXCLUDE = "        'Unnamed: 0', 'index',\n    }"
    NEW_EXCLUDE = """        'Unnamed: 0', 'index',
        'master_game_id', 'sport_code', 'home_team_name', 'away_team_name',
        'total_points', 'score_margin', 'scheduled_at',
        'moneyline_home_close', 'moneyline_away_close',
        'moneyline_home_open', 'moneyline_away_open',
        'spread_open', 'total_open', 'spread_movement', 'total_movement',
        'pinnacle_spread', 'pinnacle_total', 'pinnacle_ml_home', 'pinnacle_ml_away',
        'no_vig_home_prob', 'no_vig_away_prob',
        'consensus_spread', 'consensus_total',
    }"""

    if OLD_EXCLUDE in content and 'master_game_id' not in content.split('exclude_cols')[1].split('}')[0]:
        content = content.replace(OLD_EXCLUDE, NEW_EXCLUDE)
        patches_applied += 1
        print(f"  ✅ [FIX {patches_applied}] Expanded exclude_cols with meta/odds columns")
    elif 'master_game_id' in content.split('exclude_cols')[1].split('}')[0] if 'exclude_cols' in content else False:
        print("  ⏭️  exclude_cols already expanded")
    else:
        print("  ⏭️  Could not find exclude_cols pattern (non-critical)")

    # ========================================================================
    # WRITE PATCHED FILE
    # ========================================================================
    print("\n" + "=" * 70)
    EVALUATE_PY.write_text(content)

    size_diff = len(content) - original_len
    print(f"\n📊 Results:")
    print(f"   Patches applied: {patches_applied}")
    print(f"   Patches failed:  {len(patches_failed)}")
    print(f"   File size delta:  {'+' if size_diff >= 0 else ''}{size_diff} chars")

    if patches_failed:
        print(f"\n⚠️  Failed patches:")
        for pf in patches_failed:
            print(f"   - {pf}")

    print(f"\n{'✅ V5.3 PATCH COMPLETE' if patches_applied > 0 else '❌ NO PATCHES APPLIED'}")
    print(f"\n📋 What v5.3 fixes:")
    print(f"   A. CSV loading: Only loads main CSV, skips sub-type files")
    print(f"   B. Data filtering: Drops NaT dates + zero-score placeholder rows")
    print(f"   C. Chronological sort: Ensures proper walk-forward validation split")
    print(f"   D. Removes broken v5.2 FIX 7 diagnostic (NameError crash)")
    print(f"   E. Expands exclude_cols to properly exclude meta/odds columns")

    print(f"\n🔄 Next: Re-run evaluation:")
    print(f"   docker exec -it royaley_api python scripts/evaluate_models.py \\")
    print(f"     --models-dir /app/models --output model_scorecard.csv --verbose")

    return patches_applied


if __name__ == '__main__':
    result = main()
    sys.exit(0 if result > 0 else 1)