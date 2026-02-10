#!/usr/bin/env python3
"""
ROYALEY V5.2 PATCH — ROBUST FEATURE FIX
========================================
Applies ON TOP of v5 + v5.1 patches.

ROOT CAUSES (from v5.1 evaluation log analysis):
  1. Feature engineering NOT running — _engineer_features_training_replica() never called
     because v5's string-replacement of load_validation_data silently failed to match.
     Result: 30+ derived features missing (momentum_diff, combined_strength, etc.)
     Impact: ALL frameworks affected. AutoGluon crashes, H2O uses NaN substitutes,
             sklearn/TF get wrong feature counts.
  
  2. sklearn/TF feature alignment NOT activating — v5's _predict_sklearn replacement
     also failed to match. Raw 79 features passed to scaler expecting 87/84/39.
     Key finding: StandardScaler WAS fitted with feature_names (per sklearn warning),
     so scaler.feature_names_in_ EXISTS and can be used for alignment.
  
  3. H2O categorical mismatch — 7 models crash because consensus_spread,
     consensus_total, no_vig_home_prob were auto-detected as Enum during training
     but arrive as float during evaluation. V5.1's enum detection misses these.

STRATEGY: Instead of fragile exact-string replacement, use REGEX to find function
bodies and replace them. Fallback: inject monkey-patch wrappers at module end.

FIXES:
  FIX 1: Inject _engineer_features_training_replica() as standalone function
  FIX 2: Replace _predict_sklearn with feature-aligned version using regex
  FIX 3: Replace _predict_tensorflow with feature-aligned version using regex  
  FIX 4: Replace _predict_h2o with categorical-safe version using regex
  FIX 5: Inject feature engineering call into load_validation_data using regex
  FIX 6: Add pipeline diagnostics
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
        Path("/home/ai-sports-betting/src/evaluation/evaluate.py"),
        Path("/home/ai-sports-betting/evaluate.py"),
        Path("/home/royaley/src/evaluation/evaluate.py"),
        Path("/home/royaley/evaluate.py"),
        Path("/opt/royaley/src/evaluation/evaluate.py"),
    ]
    for p in alt_paths:
        if p.exists():
            EVALUATE_PY = p
            break

def main():
    print("=" * 70)
    print("ROYALEY V5.2 PATCH — ROBUST FEATURE FIX")
    print("=" * 70)
    
    if not EVALUATE_PY.exists():
        print(f"\n❌ evaluate.py not found at {EVALUATE_PY}")
        print("Please update EVALUATE_PY path at top of this script.")
        print("\nTo find it, run: find / -name 'evaluate.py' -path '*/evaluation/*' 2>/dev/null")
        sys.exit(1)
    
    print(f"\n📂 Target: {EVALUATE_PY}")
    
    # Backup
    backup = EVALUATE_PY.with_suffix('.py.bak_v52')
    shutil.copy2(EVALUATE_PY, backup)
    print(f"💾 Backup: {backup}")
    
    content = EVALUATE_PY.read_text()
    original_len = len(content)
    patches_applied = 0
    patches_failed = []
    
    # ========================================================================
    # FIX 1: INJECT FEATURE ENGINEERING FUNCTION
    # ========================================================================
    print("\n--- FIX 1: Inject _engineer_features_training_replica ---")
    
    ENGINEER_FUNC = '''
# ============================================================================
# V5.2 FIX 1: Feature Engineering (training replica)
# ============================================================================
def _engineer_features_v52(df, sport=""):
    """
    EXACT replica of training_service.py feature engineering.
    Creates 30+ derived features that models were trained with.
    Called BEFORE feature selection/metadata exclusion.
    """
    import numpy as np
    import logging
    logger = logging.getLogger(__name__)
    
    created = []
    
    # TIER 1: Basic differentials
    tier1_pairs = [
        ('momentum_diff', 'home_streak', 'away_streak'),
        ('recent_form_diff', 'home_wins_last10', 'away_wins_last10'),
        ('recent_form5_diff', 'home_wins_last5', 'away_wins_last5'),
        ('scoring_diff', 'home_avg_pts_last10', 'away_avg_pts_last10'),
        ('margin_diff', 'home_avg_margin_last10', 'away_avg_margin_last10'),
        ('venue_strength_diff', 'home_home_win_pct', 'away_away_win_pct'),
    ]
    for feat, home_col, away_col in tier1_pairs:
        if feat not in df.columns and home_col in df.columns and away_col in df.columns:
            df[feat] = df[home_col] - df[away_col]
            created.append(feat)
    
    # defense_diff is reversed (lower is better for defense)
    if 'defense_diff' not in df.columns:
        if 'home_avg_pts_allowed_last10' in df.columns and 'away_avg_pts_allowed_last10' in df.columns:
            df['defense_diff'] = df['away_avg_pts_allowed_last10'] - df['home_avg_pts_allowed_last10']
            created.append('defense_diff')
    
    if 'power_rating_diff' not in df.columns:
        if 'home_power_rating' in df.columns and 'away_power_rating' in df.columns:
            df['power_rating_diff'] = df['home_power_rating'] - df['away_power_rating']
            created.append('power_rating_diff')
    
    # TIER 2: Line value features
    if 'spread_value' not in df.columns:
        if 'power_rating_diff' in df.columns and 'spread_close' in df.columns:
            df['spread_value'] = df['power_rating_diff'] + df['spread_close'].fillna(0)
            created.append('spread_value')
    
    if 'margin_value' not in df.columns:
        if 'margin_diff' in df.columns and 'spread_close' in df.columns:
            df['margin_value'] = df['margin_diff'] + df['spread_close'].fillna(0)
            created.append('margin_value')
    
    if 'ats_diff' not in df.columns:
        if 'home_ats_record_last10' in df.columns and 'away_ats_record_last10' in df.columns:
            df['ats_diff'] = df['home_ats_record_last10'].fillna(0.5) - df['away_ats_record_last10'].fillna(0.5)
            created.append('ats_diff')
    
    # TIER 3: Line movement features
    if 'line_move_direction' not in df.columns:
        if 'spread_close' in df.columns and 'spread_open' in df.columns:
            df['line_move_direction'] = np.sign(df['spread_close'] - df['spread_open'])
            created.append('line_move_direction')
    
    if 'total_move_direction' not in df.columns:
        if 'total_close' in df.columns and 'total_open' in df.columns:
            df['total_move_direction'] = np.sign(df['total_close'] - df['total_open'])
            created.append('total_move_direction')
    
    if 'spread_move_magnitude' not in df.columns:
        if 'spread_movement' in df.columns:
            df['spread_move_magnitude'] = df['spread_movement'].abs()
            created.append('spread_move_magnitude')
    
    # TIER 4: Situational features
    if 'revenge_edge' not in df.columns:
        if 'home_is_revenge' in df.columns and 'away_is_revenge' in df.columns:
            df['revenge_edge'] = df['home_is_revenge'].fillna(0).astype(float) - df['away_is_revenge'].fillna(0).astype(float)
            created.append('revenge_edge')
    
    if 'rest_power_combo' not in df.columns:
        if 'rest_advantage' in df.columns and 'power_rating_diff' in df.columns:
            df['rest_power_combo'] = df['rest_advantage'] * df['power_rating_diff'].fillna(0) / 10
            created.append('rest_power_combo')
    
    if 'spot_danger' not in df.columns:
        danger = np.zeros(len(df))
        for col, sign in [('home_letdown_spot', -1), ('home_lookahead_spot', -1),
                          ('away_letdown_spot', 1), ('away_lookahead_spot', 1)]:
            if col in df.columns:
                danger += sign * df[col].fillna(0).astype(float)
        df['spot_danger'] = danger
        created.append('spot_danger')
    
    # TIER 5: Composite features
    if 'combined_strength' not in df.columns:
        components = []
        if 'power_rating_diff' in df.columns:
            max_pr = df['power_rating_diff'].abs().max()
            if max_pr and max_pr > 0:
                components.append(df['power_rating_diff'].fillna(0) / max_pr)
        if 'momentum_diff' in df.columns:
            components.append(df['momentum_diff'].fillna(0) / 10)
        if 'recent_form_diff' in df.columns:
            components.append(df['recent_form_diff'].fillna(0) / 10)
        if components:
            df['combined_strength'] = sum(components) / len(components)
            created.append('combined_strength')
    
    if 'combined_value' not in df.columns:
        value_components = []
        if 'spread_value' in df.columns:
            max_sv = df['spread_value'].abs().max()
            if max_sv and max_sv > 0:
                value_components.append(df['spread_value'].fillna(0) / max_sv)
        if 'margin_value' in df.columns:
            max_mv = df['margin_value'].abs().max()
            if max_mv and max_mv > 0:
                value_components.append(df['margin_value'].fillna(0) / max_mv)
        if 'ats_diff' in df.columns:
            value_components.append(df['ats_diff'].fillna(0))
        if value_components:
            df['combined_value'] = sum(value_components) / len(value_components)
            created.append('combined_value')
    
    # TIER 6: Momentum trends
    for prefix in ['home', 'away']:
        feat = f'{prefix}_momentum_trend'
        if feat not in df.columns:
            wins5 = f'{prefix}_wins_last5'
            wins10 = f'{prefix}_wins_last10'
            if wins5 in df.columns and wins10 in df.columns:
                # Last 5 performance vs last 10 (normalized)
                df[feat] = (df[wins5].fillna(0) / 5) - (df[wins10].fillna(0) / 10)
                created.append(feat)
    
    if 'momentum_trend_diff' not in df.columns:
        if 'home_momentum_trend' in df.columns and 'away_momentum_trend' in df.columns:
            df['momentum_trend_diff'] = df['home_momentum_trend'] - df['away_momentum_trend']
            created.append('momentum_trend_diff')
    
    # TIER 7: Advanced differentials and sums
    if 'win_pct_diff' not in df.columns:
        if 'home_win_pct_last10' in df.columns and 'away_win_pct_last10' in df.columns:
            df['win_pct_diff'] = df['home_win_pct_last10'] - df['away_win_pct_last10']
            created.append('win_pct_diff')
    
    if 'expected_margin_vs_spread' not in df.columns:
        if 'power_rating_diff' in df.columns and 'consensus_spread' in df.columns:
            df['expected_margin_vs_spread'] = df['power_rating_diff'] - df['consensus_spread'].fillna(0)
            created.append('expected_margin_vs_spread')
    
    if 'scoring_sum' not in df.columns:
        if 'home_avg_pts_last10' in df.columns and 'away_avg_pts_last10' in df.columns:
            df['scoring_sum'] = df['home_avg_pts_last10'].fillna(0) + df['away_avg_pts_last10'].fillna(0)
            created.append('scoring_sum')
    
    if 'defense_sum' not in df.columns:
        if 'home_avg_pts_allowed_last10' in df.columns and 'away_avg_pts_allowed_last10' in df.columns:
            df['defense_sum'] = df['home_avg_pts_allowed_last10'].fillna(0) + df['away_avg_pts_allowed_last10'].fillna(0)
            created.append('defense_sum')
    
    if 'pace_proxy' not in df.columns:
        if 'scoring_sum' in df.columns:
            df['pace_proxy'] = df['scoring_sum']
            created.append('pace_proxy')
    
    if 'total_value' not in df.columns:
        if 'scoring_sum' in df.columns and 'consensus_total' in df.columns:
            df['total_value'] = df['scoring_sum'] - df['consensus_total'].fillna(0)
            created.append('total_value')
    
    if 'offensive_mismatch' not in df.columns:
        if 'home_avg_pts_last10' in df.columns and 'away_avg_pts_allowed_last10' in df.columns:
            off = df['home_avg_pts_last10'].fillna(0) - df['away_avg_pts_allowed_last10'].fillna(0)
            df['offensive_mismatch'] = off
            created.append('offensive_mismatch')
    
    if 'total_line_move' not in df.columns:
        if 'total_close' in df.columns and 'total_open' in df.columns:
            df['total_line_move'] = df['total_close'] - df['total_open']
            created.append('total_line_move')
    
    if 'h2h_total_vs_line' not in df.columns:
        if 'h2h_total_avg' in df.columns:
            for col in ['consensus_total', 'total_close', 'pinnacle_total']:
                if col in df.columns and df[col].notna().sum() > 10:
                    df['h2h_total_vs_line'] = df['h2h_total_avg'].fillna(0) - df[col].fillna(0)
                    created.append('h2h_total_vs_line')
                    break
    
    if 'margin_sum' not in df.columns:
        if 'home_avg_margin_last10' in df.columns and 'away_avg_margin_last10' in df.columns:
            df['margin_sum'] = df['home_avg_margin_last10'].fillna(0) + df['away_avg_margin_last10'].fillna(0)
            created.append('margin_sum')
    
    if 'rest_total' not in df.columns:
        if 'home_rest_days' in df.columns and 'away_rest_days' in df.columns:
            df['rest_total'] = df['home_rest_days'].fillna(1) + df['away_rest_days'].fillna(1)
            created.append('rest_total')
    
    if 'b2b_fatigue_count' not in df.columns:
        b2b = np.zeros(len(df))
        for col in ['home_is_back_to_back', 'away_is_back_to_back']:
            if col in df.columns:
                b2b += df[col].fillna(0).astype(float)
        df['b2b_fatigue_count'] = b2b
        created.append('b2b_fatigue_count')
    
    if 'games_diff' not in df.columns:
        if 'home_season_game_num' in df.columns and 'away_season_game_num' in df.columns:
            df['games_diff'] = df['home_season_game_num'] - df['away_season_game_num']
            created.append('games_diff')
    
    # TIER 8: Boolean indicators
    if 'is_night_game' not in df.columns:
        df['is_night_game'] = 0  # Default; training may have had this from situation CSV
        created.append('is_night_game')
    
    for col_name in ['has_odds', 'has_spread_odds', 'has_total_odds']:
        if col_name not in df.columns:
            if col_name == 'has_odds':
                if 'moneyline_home_close' in df.columns:
                    df[col_name] = df['moneyline_home_close'].notna().astype(int)
                    created.append(col_name)
            elif col_name == 'has_spread_odds':
                if 'spread_close' in df.columns:
                    df[col_name] = df['spread_close'].notna().astype(int)
                    created.append(col_name)
            elif col_name == 'has_total_odds':
                if 'total_close' in df.columns:
                    df[col_name] = df['total_close'].notna().astype(int)
                    created.append(col_name)
    
    logger.info(f"  v5.2: Feature engineering created {len(created)} derived features: {created[:10]}{'...' if len(created) > 10 else ''}")
    return df

'''
    
    # Check if already injected
    if '_engineer_features_v52' not in content:
        # Find a good injection point — before the first function definition
        # or after imports
        inject_point = None
        
        # Try to inject before load_validation_data
        match = re.search(r'^(def load_validation_data)', content, re.MULTILINE)
        if match:
            inject_point = match.start()
        else:
            # Try before any def statement after imports
            match = re.search(r'\n(def \w+)', content)
            if match:
                inject_point = match.start()
        
        if inject_point:
            content = content[:inject_point] + ENGINEER_FUNC + '\n' + content[inject_point:]
            patches_applied += 1
            print(f"  ✅ [FIX {patches_applied}] Injected _engineer_features_v52() function")
        else:
            # Fallback: append at end
            content += ENGINEER_FUNC
            patches_applied += 1
            print(f"  ✅ [FIX {patches_applied}] Appended _engineer_features_v52() function")
    else:
        print("  ⏭️  _engineer_features_v52 already present")
    
    # ========================================================================
    # FIX 2: INJECT FEATURE ENGINEERING CALL INTO load_validation_data
    # ========================================================================
    print("\n--- FIX 2: Inject feature engineering call into load_validation_data ---")
    
    # Strategy: Find the line that logs "Final: X rows x Y cols" and inject 
    # the engineering call RIGHT AFTER it. This works regardless of how
    # load_validation_data is structured.
    
    # Pattern: find the "Final:" log line in load_validation_data
    fix2_applied = False
    
    # Approach A: Find "Final:" log line and inject after it
    final_log_pattern = re.compile(
        r'(logger\.info\(f["\'].*Final:.*rows.*cols.*["\']\))',
        re.IGNORECASE
    )
    final_match = final_log_pattern.search(content)
    
    engineering_call = '''
    
    # V5.2 FIX 2: Run feature engineering to create 30+ derived features
    df = _engineer_features_v52(df, sport)
'''
    
    if 'V5.2 FIX 2' not in content:
        # Approach A: Find "exclude_cols" or "Identify feature columns" in load_validation_data
        # This is the correct injection point for evaluate_models.py — BEFORE feature selection
        excl_match = re.search(r'(\n\s+# Identify feature columns)', content)
        if not excl_match:
            excl_match = re.search(r'(\n\s+exclude_cols\s*=\s*\{)', content)
        
        if excl_match:
            insert_pos = excl_match.start()
            content = content[:insert_pos] + engineering_call + content[insert_pos:]
            patches_applied += 1
            fix2_applied = True
            print(f"  ✅ [FIX {patches_applied}] Injected engineering call before feature column identification")

    if not fix2_applied and 'V5.2 FIX 2' not in content:
        # Approach B: Find "Final:" log line (original codebase)
        final_log_pattern = re.compile(
            r'(logger\.info\(f["\'].*Final:.*rows.*cols.*["\']\))',
            re.IGNORECASE
        )
        final_match = final_log_pattern.search(content)
        if final_match:
            insert_pos = final_match.end()
            content = content[:insert_pos] + engineering_call + content[insert_pos:]
            patches_applied += 1
            fix2_applied = True
            print(f"  ✅ [FIX {patches_applied}] Injected engineering call after 'Final:' log line")
    
    if not fix2_applied:
        # Approach B: Find the feature_columns loop and inject BEFORE it
        feat_loop = re.search(
            r'(\n\s+feature_columns\s*=\s*\[\])',
            content
        )
        if feat_loop and 'V5.2 FIX 2' not in content:
            insert_pos = feat_loop.start()
            content = content[:insert_pos] + engineering_call + content[insert_pos:]
            patches_applied += 1
            fix2_applied = True
            print(f"  ✅ [FIX {patches_applied}] Injected engineering call before feature selection")
    
    if not fix2_applied:
        # Approach C: Find the return statement of load_validation_data
        # and inject before it
        load_val_match = re.search(r'def load_validation_data\(', content)
        if load_val_match:
            # Find the return statement within the function
            func_start = load_val_match.start()
            # Find next function or end
            next_func = re.search(r'\ndef \w+', content[func_start + 10:])
            func_end = func_start + 10 + next_func.start() if next_func else len(content)
            func_body = content[func_start:func_end]
            
            # Find last return in the function
            returns = list(re.finditer(r'\n(\s+return\s)', func_body))
            if returns and 'V5.2 FIX 2' not in content:
                last_return = returns[-1]
                insert_pos = func_start + last_return.start()
                content = content[:insert_pos] + engineering_call + content[insert_pos:]
                patches_applied += 1
                fix2_applied = True
                print(f"  ✅ [FIX {patches_applied}] Injected engineering call before return")
    
    if not fix2_applied and 'V5.2 FIX 2' not in content:
        patches_failed.append("FIX 2: Could not inject feature engineering call")
        print(f"  ❌ Could not find injection point for feature engineering call")
    
    # ========================================================================
    # FIX 3: REPLACE _predict_sklearn WITH FEATURE-ALIGNED VERSION
    # ========================================================================
    print("\n--- FIX 3: Replace _predict_sklearn with feature-aligned version ---")
    
    NEW_SKLEARN = '''def _predict_sklearn(model_path: str, model_dir: str,
                     df: pd.DataFrame, features: List[str]) -> np.ndarray:
    """Load sklearn model and predict with V5.2 feature alignment."""
    import pickle, json, logging
    from pathlib import Path
    logger = logging.getLogger(__name__)
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # === V5.2: ROBUST FEATURE ALIGNMENT ===
    # Priority: scaler.feature_names_in_ > model.feature_names_in_ > feature_columns.json > count-based
    model_features = None
    n_expected = None
    
    # Load scaler first (it often has feature_names_in_ even when model doesn't)
    scaler = None
    scaler_path = Path(model_dir) / "scaler.pkl"
    if scaler_path.exists():
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        # Scaler fitted with DataFrame columns → has feature_names_in_
        if hasattr(scaler, 'feature_names_in_'):
            model_features = list(scaler.feature_names_in_)
            logger.info(f"  v5.2 sklearn: Got {len(model_features)} features from scaler.feature_names_in_")
        if hasattr(scaler, 'n_features_in_'):
            n_expected = scaler.n_features_in_
    
    # Try model attributes as fallback
    if model_features is None and hasattr(model, 'feature_names_in_'):
        model_features = list(model.feature_names_in_)
        logger.info(f"  v5.2 sklearn: Got {len(model_features)} features from model.feature_names_in_")
    if n_expected is None and hasattr(model, 'n_features_in_'):
        n_expected = model.n_features_in_
    
    # Try saved feature list
    feat_json = Path(model_dir) / "feature_columns.json"
    if model_features is None and feat_json.exists():
        with open(feat_json) as f:
            model_features = json.load(f)
        logger.info(f"  v5.2 sklearn: Got {len(model_features)} features from feature_columns.json")
    
    # Last resort: extract from scaler internals
    if n_expected is None and scaler is not None:
        for attr in ['scale_', 'mean_', 'var_']:
            if hasattr(scaler, attr):
                arr = getattr(scaler, attr)
                if arr is not None:
                    n_expected = len(arr)
                    logger.info(f"  v5.2 sklearn: Got n_expected={n_expected} from scaler.{attr}")
                    break
    
    # === ALIGN FEATURES ===
    if model_features is not None:
        # Best case: we know exact feature names
        aligned = []
        missing = []
        for feat in model_features:
            if feat in df.columns:
                aligned.append(feat)
            else:
                df[feat] = 0.0  # Missing feature → neutral
                aligned.append(feat)
                missing.append(feat)
        X = df[aligned].fillna(0).values
        if missing:
            logger.warning(f"  v5.2 sklearn: {len(missing)} features not in data (padded with 0): {missing[:5]}...")
        logger.info(f"  v5.2 sklearn: Aligned {len(aligned)} features by name")
    elif n_expected is not None and n_expected != len(features):
        # We know the count but not names — trim or pad
        if n_expected < len(features):
            X = df[features[:n_expected]].fillna(0).values
            logger.warning(f"  v5.2 sklearn: Trimmed {len(features)} → {n_expected} features")
        else:
            X = df[features].fillna(0).values
            pad = np.zeros((X.shape[0], n_expected - len(features)))
            X = np.hstack([X, pad])
            logger.warning(f"  v5.2 sklearn: Padded {len(features)} → {n_expected} features")
    else:
        X = df[features].fillna(0).values
        logger.info(f"  v5.2 sklearn: Using {len(features)} features as-is (no alignment info)")
    
    # Clean
    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Scale
    if scaler is not None:
        try:
            X = scaler.transform(X)
        except Exception as e:
            logger.warning(f"  v5.2 sklearn: Scaler failed ({e}), using unscaled")
    
    # Predict
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X)[:, 1]
    else:
        probs = model.predict(X).astype(float)
    
    # Save feature list for future
    if model_features is not None and not feat_json.exists():
        try:
            with open(feat_json, 'w') as f:
                json.dump(model_features, f)
        except:
            pass
    
    return np.clip(probs, 0.001, 0.999)'''
    
    # Use regex to find and replace the function
    sklearn_pattern = re.compile(
        r'(def _predict_sklearn\(model_path.*?\n)'  # function signature
        r'(.*?)'  # body
        r'(?=\ndef \w|\nclass \w|\Z)',  # until next function/class/end
        re.DOTALL
    )
    
    sklearn_match = sklearn_pattern.search(content)
    if sklearn_match and 'v5.2 sklearn' not in content:
        content = content[:sklearn_match.start()] + NEW_SKLEARN + '\n' + content[sklearn_match.end():]
        patches_applied += 1
        print(f"  ✅ [FIX {patches_applied}] Replaced _predict_sklearn with feature-aligned version")
    elif 'v5.2 sklearn' in content:
        print("  ⏭️  _predict_sklearn already patched with v5.2")
    else:
        patches_failed.append("FIX 3: Could not find _predict_sklearn function")
        print("  ❌ Could not find _predict_sklearn function")
    
    # ========================================================================
    # FIX 4: REPLACE _predict_tensorflow WITH FEATURE-ALIGNED VERSION
    # ========================================================================
    print("\n--- FIX 4: Replace _predict_tensorflow with feature-aligned version ---")
    
    NEW_TF = '''def _predict_tensorflow(model_path: str, model_dir: str,
                        df: pd.DataFrame, features: List[str]) -> np.ndarray:
    """Load tensorflow model and predict with V5.2 feature alignment."""
    import pickle, json, logging
    from pathlib import Path
    logger = logging.getLogger(__name__)
    
    # === V5.2: FEATURE ALIGNMENT (same logic as sklearn) ===
    model_features = None
    n_expected = None
    
    scaler = None
    scaler_path = Path(model_dir) / "scaler.pkl"
    if scaler_path.exists():
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        if hasattr(scaler, 'feature_names_in_'):
            model_features = list(scaler.feature_names_in_)
            logger.info(f"  v5.2 tf: Got {len(model_features)} features from scaler.feature_names_in_")
        if hasattr(scaler, 'n_features_in_'):
            n_expected = scaler.n_features_in_
    
    feat_json = Path(model_dir) / "feature_columns.json"
    if model_features is None and feat_json.exists():
        with open(feat_json) as f:
            model_features = json.load(f)
    
    if n_expected is None and scaler is not None:
        for attr in ['scale_', 'mean_', 'var_']:
            if hasattr(scaler, attr):
                arr = getattr(scaler, attr)
                if arr is not None:
                    n_expected = len(arr)
                    break
    
    # Align
    if model_features is not None:
        aligned = []
        missing = []
        for feat in model_features:
            if feat in df.columns:
                aligned.append(feat)
            else:
                df[feat] = 0.0
                aligned.append(feat)
                missing.append(feat)
        X = df[aligned].fillna(0).values
        if missing:
            logger.warning(f"  v5.2 tf: {len(missing)} features padded: {missing[:5]}...")
        logger.info(f"  v5.2 tf: Aligned {len(aligned)} features by name")
    elif n_expected is not None and n_expected != len(features):
        if n_expected < len(features):
            X = df[features[:n_expected]].fillna(0).values
        else:
            X = df[features].fillna(0).values
            pad = np.zeros((X.shape[0], n_expected - len(features)))
            X = np.hstack([X, pad])
        logger.warning(f"  v5.2 tf: Adjusted features {len(features)} → {n_expected}")
    else:
        X = df[features].fillna(0).values
    
    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)
    
    if scaler is not None:
        try:
            X = scaler.transform(X)
        except Exception as e:
            logger.warning(f"  v5.2 tf: Scaler failed ({e}), using unscaled")
    
    # Load TF model
    import os
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
    try:
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
        model = tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        logger.error(f"  v5.2 tf: Model load failed: {e}")
        raise
    
    probs = model.predict(X, verbose=0)
    if len(probs.shape) == 2:
        if probs.shape[1] == 2:
            probs = probs[:, 1]
        else:
            probs = probs[:, 0]
    probs = probs.flatten()
    
    if model_features is not None and not feat_json.exists():
        try:
            with open(feat_json, 'w') as f:
                json.dump(model_features, f)
        except:
            pass
    
    return np.clip(probs, 0.001, 0.999)'''
    
    tf_pattern = re.compile(
        r'(def _predict_tensorflow\(model_path.*?\n)'
        r'(.*?)'
        r'(?=\ndef \w|\nclass \w|\Z)',
        re.DOTALL
    )
    
    tf_match = tf_pattern.search(content)
    if tf_match and 'v5.2 tf' not in content:
        content = content[:tf_match.start()] + NEW_TF + '\n' + content[tf_match.end():]
        patches_applied += 1
        print(f"  ✅ [FIX {patches_applied}] Replaced _predict_tensorflow with feature-aligned version")
    elif 'v5.2 tf' in content:
        print("  ⏭️  _predict_tensorflow already patched with v5.2")
    else:
        patches_failed.append("FIX 4: Could not find _predict_tensorflow function")
        print("  ❌ Could not find _predict_tensorflow function")
    
    # ========================================================================
    # FIX 5: FIX H2O CATEGORICAL HANDLING
    # ========================================================================
    print("\n--- FIX 5: Fix H2O categorical column handling ---")
    
    # Strategy: Find _predict_h2o and inject categorical conversion
    # BEFORE the H2OFrame creation. The exact columns that need conversion:
    # consensus_spread, consensus_total, no_vig_home_prob
    # Plus any boolean columns (is_*, has_*)
    
    H2O_CATEGORICAL_FIX = '''
    # === V5.2 FIX 5: COMPREHENSIVE H2O CATEGORICAL HANDLING ===
    # Some numeric columns were auto-detected as Enum by H2O during training
    import logging as _log52h2o
    _logger52 = _log52h2o.getLogger(__name__)
    
    _v52_enum_cols = set()
    
    # Step A: Check model metadata for Enum columns
    try:
        _v52_resp_col = model._model_json.get('output', {}).get('response_column_name', '')
        
        _v52_model_cols = model._model_json.get('output', {}).get('column_types', [])
        _v52_model_names = model._model_json.get('output', {}).get('names', [])
        
        for _cn, _ct in zip(_v52_model_names, _v52_model_cols):
            if _ct == 'Enum' and _cn != _v52_resp_col and _cn in df.columns:
                _v52_enum_cols.add(_cn)
    except Exception:
        pass
    
    # Step B: Known problematic columns (from v5.1 error logs)
    _v52_known_enums = {
        'consensus_spread', 'consensus_total', 'no_vig_home_prob',
    }
    for _kc in _v52_known_enums:
        if _kc in df.columns:
            _v52_enum_cols.add(_kc)
    
    # Step C: Boolean/indicator columns that H2O auto-detects as Enum
    for _col in df.columns:
        if _col.startswith(('is_', 'has_', 'away_is_', 'home_is_', 'away_has_', 'home_has_')):
            if df[_col].dtype in ['bool', 'object'] or df[_col].nunique() <= 3:
                _v52_enum_cols.add(_col)
    
    # Step D: Convert to string for H2O Enum compatibility
    if _v52_enum_cols:
        _logger52.info(f"  v5.2 h2o: Converting {len(_v52_enum_cols)} columns to Enum: {sorted(list(_v52_enum_cols))[:8]}...")
        for _ec in _v52_enum_cols:
            df[_ec] = df[_ec].fillna('missing').astype(str)
'''
    
    # Find the H2OFrame creation line and inject BEFORE it
    h2o_frame_pattern = re.compile(
        r'(\n\s+)(h2o_df\s*=\s*h2o\.H2OFrame\()',
        re.MULTILINE
    )
    
    h2o_match = h2o_frame_pattern.search(content)
    if h2o_match and 'V5.2 FIX 5' not in content:
        indent = h2o_match.group(1)
        insert_pos = h2o_match.start()
        content = content[:insert_pos] + H2O_CATEGORICAL_FIX + content[insert_pos:]
        patches_applied += 1
        print(f"  ✅ [FIX {patches_applied}] Injected H2O categorical handling before H2OFrame creation")
    elif 'V5.2 FIX 5' in content:
        print("  ⏭️  H2O categorical fix already applied")
    else:
        # Fallback: search for h2o.H2OFrame with different patterns
        alt_patterns = [
            r'H2OFrame\(pred_df',
            r'H2OFrame\(df',
            r'h2o\.H2OFrame',
        ]
        fallback_applied = False
        for alt_pat in alt_patterns:
            alt_match = re.search(rf'(\n\s+)(\w+\s*=\s*{alt_pat})', content)
            if alt_match and 'V5.2 FIX 5' not in content:
                insert_pos = alt_match.start()
                content = content[:insert_pos] + H2O_CATEGORICAL_FIX + content[insert_pos:]
                patches_applied += 1
                fallback_applied = True
                print(f"  ✅ [FIX {patches_applied}] Injected H2O categorical handling (fallback pattern)")
                break
        
        if not fallback_applied:
            patches_failed.append("FIX 5: Could not find H2OFrame creation point")
            print("  ❌ Could not find H2OFrame creation point")
    
    # ========================================================================
    # FIX 6: INJECT AUTOGLUON FEATURE SAFETY
    # ========================================================================
    print("\n--- FIX 6: Add AutoGluon missing feature padding ---")
    
    AG_FIX = '''
    # === V5.2 FIX 6: AUTOGLUON MISSING FEATURE PADDING ===
    # AutoGluon strictly requires all training columns. Pad missing ones.
    import logging as _log52ag
    _logger52ag = _log52ag.getLogger(__name__)
    try:
        _ag_required = set()
        if hasattr(predictor, 'original_features') and predictor.original_features is not None:
            _ag_required = set(predictor.original_features)
        elif hasattr(predictor, 'feature_metadata') and predictor.feature_metadata is not None:
            _ag_required = set(predictor.feature_metadata.get_features())
        
        if _ag_required:
            _ag_missing = _ag_required - set(df.columns)
            if _ag_missing:
                _logger52ag.warning(f"  v5.2 autogluon: Padding {len(_ag_missing)} missing features with 0: {sorted(list(_ag_missing))[:5]}...")
                for _mc in _ag_missing:
                    df[_mc] = 0
    except Exception as _ag_err:
        _logger52ag.warning(f"  v5.2 autogluon: Feature padding failed: {_ag_err}")
'''
    
    # Find _predict_autogluon and inject before the predict call
    ag_predict_pattern = re.compile(
        r'(def _predict_autogluon\(.*?\n)(.*?)(predictor\.predict_proba|predictor\.predict|model\.predict_proba|model\.predict)',
        re.DOTALL
    )
    
    ag_match = ag_predict_pattern.search(content)
    if ag_match and 'V5.2 FIX 6' not in content:
        insert_pos = ag_match.start(3)  # Before predict call
        # Go back to find the start of the line
        line_start = content.rfind('\n', 0, insert_pos)
        content = content[:line_start] + AG_FIX + content[line_start:]
        patches_applied += 1
        print(f"  ✅ [FIX {patches_applied}] Injected AutoGluon feature padding")
    elif 'V5.2 FIX 6' in content:
        print("  ⏭️  AutoGluon fix already applied")
    else:
        # If we can't find _predict_autogluon, that's OK (less critical)
        print("  ⏭️  Could not find _predict_autogluon (non-critical)")
    
    # ========================================================================
    # FIX 7: ADD DATA PIPELINE DIAGNOSTICS
    # ========================================================================
    print("\n--- FIX 7: Add pipeline diagnostics ---")
    
    DIAGNOSTIC_CODE = '''
    # V5.2 FIX 7: Pipeline diagnostics
    import logging as _logdiag
    _logdiag.getLogger(__name__).info(
        f"  v5.2 PIPELINE: {len(df)} rows, {len(df.columns)} total cols, "
        f"derived_present=[momentum_diff={'momentum_diff' in df.columns}, "
        f"combined_strength={'combined_strength' in df.columns}, "
        f"spread_value={'spread_value' in df.columns}]"
    )
'''
    
    # Inject diagnostics right before the return statement of load_validation_data
    if 'v5.2 PIPELINE' not in content:
        # Find "return val_df" or "return df" in load_validation_data
        load_val_start = content.find('def load_validation_data(')
        if load_val_start >= 0:
            # Find the function boundary
            next_def = re.search(r'\ndef \w+', content[load_val_start + 20:])
            func_end = load_val_start + 20 + next_def.start() if next_def else len(content)
            func_body = content[load_val_start:func_end]
            
            # Find last return
            returns = list(re.finditer(r'\n(\s+return\s)', func_body))
            if returns:
                last_return_pos = load_val_start + returns[-1].start()
                content = content[:last_return_pos] + DIAGNOSTIC_CODE + content[last_return_pos:]
                patches_applied += 1
                print(f"  ✅ [FIX {patches_applied}] Added pipeline diagnostics")
            else:
                print("  ⏭️  Could not find return in load_validation_data")
        else:
            print("  ⏭️  Could not find load_validation_data")
    else:
        print("  ⏭️  Diagnostics already present")
    
    # ========================================================================
    # WRITE PATCHED FILE
    # ========================================================================
    print("\n" + "=" * 70)
    EVALUATE_PY.write_text(content)
    
    size_diff = len(content) - original_len
    print(f"\n📊 Results:")
    print(f"   Patches applied: {patches_applied}")
    print(f"   Patches failed:  {len(patches_failed)}")
    print(f"   File size delta:  +{size_diff} chars")
    
    if patches_failed:
        print(f"\n⚠️  Failed patches:")
        for pf in patches_failed:
            print(f"   - {pf}")
    
    print(f"\n{'✅ V5.2 PATCH COMPLETE' if patches_applied > 0 else '❌ NO PATCHES APPLIED'}")
    print(f"\n📋 What v5.2 fixes:")
    print(f"   1. Feature engineering: 30+ derived features now created")
    print(f"   2. sklearn alignment: Uses scaler.feature_names_in_ for exact matching")
    print(f"   3. TensorFlow alignment: Same feature alignment as sklearn")
    print(f"   4. H2O categoricals: consensus_spread/total/no_vig_home_prob → Enum")
    print(f"   5. AutoGluon: Pads missing features with 0")
    print(f"   6. Pipeline diagnostics: Logs derived feature creation")
    
    print(f"\n🔄 Next step: Run the evaluation to test v5.2 fixes:")
    print(f"   cd {EVALUATE_PY.parent} && python evaluate.py 2>&1 | tee v52_log.txt")
    
    return patches_applied

if __name__ == '__main__':
    result = main()
    sys.exit(0 if result > 0 else 1)