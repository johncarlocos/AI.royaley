#!/usr/bin/env python3
"""
ROYALEY - Evaluation Patch (COMBINED v3 + v4)
==============================================
Single unified patch that applies ALL fixes at once.

v3 Fixes (Feature Engineering & Model Loading):
  1. Missing 30+ derived features during evaluation (feature engineering)
  2. H2O categorical column mismatch (consensus_spread etc -> .asfactor())
  3. Single-class validation splits (ATP/WTA/NFL moneyline -> stratified fallback)
  4. sklearn/tensorflow feature count mismatches (pad/trim alignment)
  5. AutoGluon missing column errors (auto-add missing features)
  6. exclude_cols removing spread_close/total_close needed as input features
  7. DataFrame fragmentation warnings (df.copy() consolidation)

v4 Fixes (Scoring & Display):
  1. Suppress PerformanceWarnings during feature engineering
  2. Tiered pass system: PRODUCTION / SIGNAL / LEAKAGE_SUSPECT
  3. Fix composite score (ROI normalization, leakage penalty)
  4. Add data leakage detection (tennis home=winner bias)
  5. Better summary showing models with real signal

Usage:
    # Copy into container and run:
    docker cp patch_evaluation.py royaley_api:/app/scripts/
    docker exec royaley_api python /app/scripts/patch_evaluation.py

    # Then re-run evaluation:
    docker exec -it royaley_api python scripts/evaluate_models.py --verbose
"""

import sys
from pathlib import Path

EVAL_SCRIPT = Path("/app/scripts/evaluate_models.py")

# ============================================================================
# PATCH 1: engineer_features() function
# Exact replica of training_service.py FIX 2d feature engineering
# ============================================================================

ENGINEER_FEATURES_FUNC = '''

def engineer_features(df: pd.DataFrame, sport: str = "") -> pd.DataFrame:
    """
    Replicate the exact feature engineering from training_service.py.
    Must be called BEFORE feature column selection so models see the
    same columns they were trained on.
    
    Creates 30+ derived features matching what _prepare_training_data() builds.
    """
    derived = []

    # ── TIER 1: BASIC DIFFERENTIALS ──

    if 'power_rating_diff' not in df.columns:
        if 'home_power_rating' in df.columns and 'away_power_rating' in df.columns:
            df['power_rating_diff'] = df['home_power_rating'] - df['away_power_rating']
            derived.append('power_rating_diff')

    if 'momentum_diff' not in df.columns:
        if 'home_streak' in df.columns and 'away_streak' in df.columns:
            df['momentum_diff'] = df['home_streak'] - df['away_streak']
            derived.append('momentum_diff')

    if 'recent_form_diff' not in df.columns:
        if 'home_wins_last10' in df.columns and 'away_wins_last10' in df.columns:
            df['recent_form_diff'] = df['home_wins_last10'] - df['away_wins_last10']
            derived.append('recent_form_diff')

    if 'recent_form5_diff' not in df.columns:
        if 'home_wins_last5' in df.columns and 'away_wins_last5' in df.columns:
            df['recent_form5_diff'] = df['home_wins_last5'] - df['away_wins_last5']
            derived.append('recent_form5_diff')

    if 'scoring_diff' not in df.columns:
        if 'home_avg_pts_last10' in df.columns and 'away_avg_pts_last10' in df.columns:
            df['scoring_diff'] = df['home_avg_pts_last10'] - df['away_avg_pts_last10']
            derived.append('scoring_diff')

    if 'defense_diff' not in df.columns:
        if 'home_avg_pts_allowed_last10' in df.columns and 'away_avg_pts_allowed_last10' in df.columns:
            df['defense_diff'] = df['away_avg_pts_allowed_last10'] - df['home_avg_pts_allowed_last10']
            derived.append('defense_diff')

    if 'margin_diff' not in df.columns:
        if 'home_avg_margin_last10' in df.columns and 'away_avg_margin_last10' in df.columns:
            df['margin_diff'] = df['home_avg_margin_last10'] - df['away_avg_margin_last10']
            derived.append('margin_diff')

    if 'venue_strength_diff' not in df.columns:
        if 'home_home_win_pct' in df.columns and 'away_away_win_pct' in df.columns:
            df['venue_strength_diff'] = df['home_home_win_pct'] - df['away_away_win_pct']
            derived.append('venue_strength_diff')

    # ── TIER 2: LINE VALUE FEATURES ──

    if 'spread_value' not in df.columns:
        if 'power_rating_diff' in df.columns and 'spread_close' in df.columns:
            df['spread_value'] = df['power_rating_diff'] + df['spread_close'].fillna(0)
            derived.append('spread_value')

    if 'margin_value' not in df.columns:
        if 'margin_diff' in df.columns and 'spread_close' in df.columns:
            df['margin_value'] = df['margin_diff'] + df['spread_close'].fillna(0)
            derived.append('margin_value')

    if 'ats_diff' not in df.columns:
        if 'home_ats_record_last10' in df.columns and 'away_ats_record_last10' in df.columns:
            df['ats_diff'] = df['home_ats_record_last10'].fillna(0.5) - df['away_ats_record_last10'].fillna(0.5)
            derived.append('ats_diff')

    # ── TIER 3: LINE MOVEMENT ──

    if 'line_move_direction' not in df.columns:
        if 'spread_close' in df.columns and 'spread_open' in df.columns:
            df['line_move_direction'] = np.sign(df['spread_close'] - df['spread_open'])
            derived.append('line_move_direction')

    if 'total_move_direction' not in df.columns:
        if 'total_close' in df.columns and 'total_open' in df.columns:
            df['total_move_direction'] = np.sign(df['total_close'] - df['total_open'])
            derived.append('total_move_direction')

    if 'spread_move_magnitude' not in df.columns:
        if 'spread_movement' in df.columns:
            df['spread_move_magnitude'] = df['spread_movement'].abs()
            derived.append('spread_move_magnitude')

    # ── TIER 4: SITUATIONAL SPOTS ──

    if 'revenge_edge' not in df.columns:
        if 'home_is_revenge' in df.columns and 'away_is_revenge' in df.columns:
            df['revenge_edge'] = df['home_is_revenge'].fillna(0).astype(int) - df['away_is_revenge'].fillna(0).astype(int)
            derived.append('revenge_edge')

    if 'rest_power_combo' not in df.columns:
        if 'rest_advantage' in df.columns and 'power_rating_diff' in df.columns:
            df['rest_power_combo'] = df['rest_advantage'] * df['power_rating_diff'].fillna(0) / 10
            derived.append('rest_power_combo')

    if 'spot_danger' not in df.columns:
        danger = np.zeros(len(df))
        if 'home_letdown_spot' in df.columns:
            danger -= df['home_letdown_spot'].fillna(0).astype(int)
        if 'home_lookahead_spot' in df.columns:
            danger -= df['home_lookahead_spot'].fillna(0).astype(int)
        if 'away_letdown_spot' in df.columns:
            danger += df['away_letdown_spot'].fillna(0).astype(int)
        if 'away_lookahead_spot' in df.columns:
            danger += df['away_lookahead_spot'].fillna(0).astype(int)
        df['spot_danger'] = danger
        derived.append('spot_danger')

    # ── TIER 5: COMPOSITES ──

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
            derived.append('combined_strength')

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
            derived.append('combined_value')

    if 'home_momentum_trend' not in df.columns:
        if 'home_wins_last5' in df.columns and 'home_wins_last10' in df.columns:
            df['home_momentum_trend'] = df['home_wins_last5'] - (df['home_wins_last10'] / 2)
            derived.append('home_momentum_trend')

    if 'away_momentum_trend' not in df.columns:
        if 'away_wins_last5' in df.columns and 'away_wins_last10' in df.columns:
            df['away_momentum_trend'] = df['away_wins_last5'] - (df['away_wins_last10'] / 2)
            derived.append('away_momentum_trend')

    if 'momentum_trend_diff' not in df.columns:
        if 'home_momentum_trend' in df.columns and 'away_momentum_trend' in df.columns:
            df['momentum_trend_diff'] = df['home_momentum_trend'] - df['away_momentum_trend']
            derived.append('momentum_trend_diff')

    if 'win_pct_diff' not in df.columns:
        if 'home_win_pct_last10' in df.columns and 'away_win_pct_last10' in df.columns:
            df['win_pct_diff'] = df['home_win_pct_last10'] - df['away_win_pct_last10']
            derived.append('win_pct_diff')

    if 'expected_margin_vs_spread' not in df.columns:
        if 'margin_diff' in df.columns and 'spread_close' in df.columns:
            df['expected_margin_vs_spread'] = df['margin_diff'] + df['spread_close'].fillna(0)
            derived.append('expected_margin_vs_spread')

    # ── TIER 6: TOTAL-SPECIFIC (SUM not DIFF) ──

    if 'scoring_sum' not in df.columns:
        if 'home_avg_pts_last10' in df.columns and 'away_avg_pts_last10' in df.columns:
            df['scoring_sum'] = df['home_avg_pts_last10'].fillna(0) + df['away_avg_pts_last10'].fillna(0)
            derived.append('scoring_sum')

    if 'defense_sum' not in df.columns:
        if 'home_avg_pts_allowed_last10' in df.columns and 'away_avg_pts_allowed_last10' in df.columns:
            df['defense_sum'] = df['home_avg_pts_allowed_last10'].fillna(0) + df['away_avg_pts_allowed_last10'].fillna(0)
            derived.append('defense_sum')

    if 'pace_proxy' not in df.columns:
        if 'scoring_sum' in df.columns and 'defense_sum' in df.columns:
            df['pace_proxy'] = (df['scoring_sum'] + df['defense_sum']) / 2
            derived.append('pace_proxy')

    if 'total_value' not in df.columns:
        total_line_col = None
        for col in ['consensus_total', 'total_close', 'pinnacle_total']:
            if col in df.columns and df[col].notna().sum() > 5:
                total_line_col = col
                break
        if total_line_col and 'pace_proxy' in df.columns:
            df['total_value'] = df[total_line_col].fillna(df['pace_proxy']) - df['pace_proxy']
            derived.append('total_value')

    if 'offensive_mismatch' not in df.columns:
        home_off = df.get('home_avg_pts_last10')
        away_def = df.get('away_avg_pts_allowed_last10')
        away_off = df.get('away_avg_pts_last10')
        home_def = df.get('home_avg_pts_allowed_last10')
        if home_off is not None and away_def is not None and away_off is not None and home_def is not None:
            home_exploit = home_off.fillna(0) - away_def.fillna(0)
            away_exploit = away_off.fillna(0) - home_def.fillna(0)
            df['offensive_mismatch'] = home_exploit + away_exploit
            derived.append('offensive_mismatch')

    if 'total_line_move' not in df.columns:
        if 'total_close' in df.columns and 'total_open' in df.columns:
            df['total_line_move'] = df['total_close'] - df['total_open']
            derived.append('total_line_move')

    if 'h2h_total_vs_line' not in df.columns:
        if 'h2h_total_avg' in df.columns:
            total_line_col = None
            for col in ['consensus_total', 'total_close', 'pinnacle_total']:
                if col in df.columns and df[col].notna().sum() > 5:
                    total_line_col = col
                    break
            if total_line_col:
                df['h2h_total_vs_line'] = df['h2h_total_avg'].fillna(0) - df[total_line_col].fillna(0)
                derived.append('h2h_total_vs_line')

    if 'margin_sum' not in df.columns:
        if 'home_avg_margin_last10' in df.columns and 'away_avg_margin_last10' in df.columns:
            df['margin_sum'] = df['home_avg_margin_last10'].fillna(0) + df['away_avg_margin_last10'].fillna(0)
            derived.append('margin_sum')

    if 'rest_total' not in df.columns:
        if 'home_rest_days' in df.columns and 'away_rest_days' in df.columns:
            df['rest_total'] = df['home_rest_days'].fillna(1) + df['away_rest_days'].fillna(1)
            derived.append('rest_total')

    if 'b2b_fatigue_count' not in df.columns:
        b2b_count = np.zeros(len(df))
        if 'home_is_back_to_back' in df.columns:
            b2b_count += df['home_is_back_to_back'].fillna(0).astype(int)
        if 'away_is_back_to_back' in df.columns:
            b2b_count += df['away_is_back_to_back'].fillna(0).astype(int)
        df['b2b_fatigue_count'] = b2b_count
        derived.append('b2b_fatigue_count')

    # ── MISSINGNESS INDICATORS ──

    odds_indicator_cols = [
        'implied_home_prob', 'no_vig_home_prob',
        'moneyline_home_close', 'moneyline_away_close',
        'spread_close', 'total_close',
    ]
    if 'has_odds' not in df.columns:
        odds_present = pd.Series(False, index=df.index)
        for col in odds_indicator_cols:
            if col in df.columns:
                odds_present = odds_present | df[col].notna()
        if odds_present.any():
            df['has_odds'] = odds_present.astype(int)
            derived.append('has_odds')

    if 'has_spread_odds' not in df.columns:
        spread_cols = [c for c in ['spread_close', 'spread_open'] if c in df.columns]
        if spread_cols:
            df['has_spread_odds'] = df[spread_cols].notna().any(axis=1).astype(int)
            derived.append('has_spread_odds')

    if 'has_total_odds' not in df.columns:
        total_cols = [c for c in ['total_close', 'total_open'] if c in df.columns]
        if total_cols:
            df['has_total_odds'] = df[total_cols].notna().any(axis=1).astype(int)
            derived.append('has_total_odds')

    # ── FALLBACK LINE INDICATORS ──

    if 'has_real_spread_line' not in df.columns:
        for scol in ['spread_close', 'spread_line', 'home_spread']:
            if scol in df.columns:
                df['has_real_spread_line'] = df[scol].notna().astype(int)
                derived.append('has_real_spread_line')
                break
        if 'has_real_spread_line' not in df.columns:
            df['has_real_spread_line'] = 0
            derived.append('has_real_spread_line')

    if 'has_real_total_line' not in df.columns:
        for tcol in ['total_close', 'total_line', 'over_under_line']:
            if tcol in df.columns:
                df['has_real_total_line'] = df[tcol].notna().astype(int)
                derived.append('has_real_total_line')
                break
        if 'has_real_total_line' not in df.columns:
            df['has_real_total_line'] = 0
            derived.append('has_real_total_line')

    # ── BOOLEAN DEFAULTS (columns models may expect) ──

    for col in ['away_is_revenge', 'home_is_revenge',
                'away_letdown_spot', 'home_letdown_spot',
                'away_3_in_4_nights', 'home_3_in_4_nights',
                'away_is_back_to_back', 'home_is_back_to_back',
                'is_night_game']:
        if col not in df.columns:
            df[col] = 0
            derived.append(col)

    if derived:
        logger.info(
            f"  \\u2728 Feature engineering: created {len(derived)} derived features "
            f"({derived[:8]}{'...' if len(derived) > 8 else ''})"
        )

    # Defragment DataFrame after many column insertions (eliminates PerformanceWarning)
    return df.copy()

'''

# ============================================================================
# PATCH 2: Updated load_and_predict with feature alignment
# ============================================================================

OLD_LOAD_AND_PREDICT = '''def load_and_predict(model_info: Dict, validation_df: pd.DataFrame,
                     feature_columns: List[str]) -> Optional[np.ndarray]:
    """Load a model and generate predictions on validation data."""
    framework = model_info["framework"]
    model_path = model_info["model_path"]

    try:
        if framework == "sklearn":
            return _predict_sklearn(model_path, model_info["model_dir"],
                                    validation_df, feature_columns)
        elif framework == "tensorflow":
            return _predict_tensorflow(model_path, model_info["model_dir"],
                                       validation_df, feature_columns)
        elif framework == "h2o":
            return _predict_h2o(model_path, model_info["model_dir"],
                                validation_df, feature_columns)
        elif framework == "autogluon":
            return _predict_autogluon(model_path, validation_df, feature_columns)
        elif framework == "quantum":
            return _predict_quantum(model_path, model_info["model_dir"],
                                    validation_df, feature_columns)
        elif framework == "meta_ensemble":
            return _predict_meta_ensemble(model_path, model_info["model_dir"],
                                          validation_df, feature_columns)
    except Exception as e:
        logger.error(f"Prediction failed for {framework}/{model_info['sport']}/{model_info['bet_type']}: {e}")
        return None

    return None'''

NEW_LOAD_AND_PREDICT = '''def load_and_predict(model_info: Dict, validation_df: pd.DataFrame,
                     feature_columns: List[str]) -> Optional[np.ndarray]:
    """Load a model and generate predictions on validation data.
    
    Handles feature alignment: if model expects features not in validation_df,
    adds them as zeros. If validation has extra features, they are ignored
    by the model's own feature selection.
    """
    framework = model_info["framework"]
    model_path = model_info["model_path"]

    try:
        if framework == "sklearn":
            return _predict_sklearn(model_path, model_info["model_dir"],
                                    validation_df, feature_columns)
        elif framework == "tensorflow":
            return _predict_tensorflow(model_path, model_info["model_dir"],
                                       validation_df, feature_columns)
        elif framework == "h2o":
            return _predict_h2o(model_path, model_info["model_dir"],
                                validation_df, feature_columns)
        elif framework == "autogluon":
            return _predict_autogluon(model_path, validation_df, feature_columns)
        elif framework == "quantum":
            return _predict_quantum(model_path, model_info["model_dir"],
                                    validation_df, feature_columns)
        elif framework == "meta_ensemble":
            return _predict_meta_ensemble(model_path, model_info["model_dir"],
                                          validation_df, feature_columns)
    except Exception as e:
        logger.error(f"Prediction failed for {framework}/{model_info['sport']}/{model_info['bet_type']}: {e}")
        logger.error(f"  Features available: {len(feature_columns)}, DataFrame cols: {len(validation_df.columns)}")
        return None

    return None'''

# ============================================================================
# PATCH 3: Updated _predict_sklearn with feature alignment
# ============================================================================

OLD_PREDICT_SKLEARN = '''def _predict_sklearn(model_path: str, model_dir: str,
                     df: pd.DataFrame, features: List[str]) -> np.ndarray:
    """Load sklearn model and predict."""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    scaler_path = Path(model_dir) / "scaler.pkl"
    X = df[features].values

    if scaler_path.exists():
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        X = scaler.transform(X)

    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X)[:, 1]
    else:
        probs = model.predict(X)

    return np.clip(probs, 0.001, 0.999)'''

NEW_PREDICT_SKLEARN = '''def _predict_sklearn(model_path: str, model_dir: str,
                     df: pd.DataFrame, features: List[str]) -> np.ndarray:
    """Load sklearn model and predict with feature alignment."""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    scaler_path = Path(model_dir) / "scaler.pkl"

    # Load feature list saved during training (if available)
    feature_list_path = Path(model_dir) / "feature_columns.json"
    train_features = None
    if feature_list_path.exists():
        import json as _json
        with open(feature_list_path) as f:
            train_features = _json.load(f)

    # Determine expected feature count from scaler or model
    expected_n = None
    scaler = None
    if scaler_path.exists():
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        if hasattr(scaler, 'n_features_in_'):
            expected_n = scaler.n_features_in_
        elif hasattr(scaler, 'mean_') and scaler.mean_ is not None:
            expected_n = len(scaler.mean_)

    if expected_n is None and hasattr(model, 'n_features_in_'):
        expected_n = model.n_features_in_

    # Align features: use saved feature list if available, otherwise pad/trim
    if train_features and expected_n and len(train_features) == expected_n:
        # Best case: exact feature list from training
        aligned_df = pd.DataFrame(0.0, index=df.index, columns=train_features)
        common = [c for c in train_features if c in df.columns]
        aligned_df[common] = df[common].values
        X = aligned_df.values.astype(np.float64)
        n_missing = len(train_features) - len(common)
        if n_missing > 0:
            missing = [c for c in train_features if c not in df.columns]
            logger.warning(f"  sklearn: {n_missing} features missing, filled with 0: {missing[:5]}...")
    elif expected_n and len(features) != expected_n:
        # Fallback: pad with zeros or trim
        X = df[features].values.astype(np.float64)
        if len(features) < expected_n:
            pad = np.zeros((len(df), expected_n - len(features)))
            X = np.hstack([X, pad])
            logger.warning(f"  sklearn: padded {expected_n - len(features)} missing features with 0")
        elif len(features) > expected_n:
            X = X[:, :expected_n]
            logger.warning(f"  sklearn: trimmed {len(features) - expected_n} extra features")
    else:
        X = df[features].values.astype(np.float64)

    # Replace NaN/inf before scaling
    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)

    if scaler is not None:
        X = scaler.transform(X)

    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X)[:, 1]
    else:
        probs = model.predict(X)

    return np.clip(probs, 0.001, 0.999)'''

# ============================================================================
# PATCH 4: Updated _predict_tensorflow with feature alignment
# ============================================================================

OLD_PREDICT_TF = '''def _predict_tensorflow(model_path: str, model_dir: str,
                        df: pd.DataFrame, features: List[str]) -> np.ndarray:
    """Load TensorFlow model and predict."""
    import tensorflow as tf

    model = tf.keras.models.load_model(model_path)

    scaler_path = Path(model_dir) / "scaler.pkl"
    X = df[features].values.astype(np.float32)

    if scaler_path.exists():
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        X = scaler.transform(X)

    # Replace NaN/inf
    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)

    probs = model.predict(X, verbose=0)
    if probs.ndim == 2 and probs.shape[1] == 1:
        probs = probs.ravel()
    elif probs.ndim == 2 and probs.shape[1] == 2:
        probs = probs[:, 1]

    return np.clip(probs, 0.001, 0.999)'''

NEW_PREDICT_TF = '''def _predict_tensorflow(model_path: str, model_dir: str,
                        df: pd.DataFrame, features: List[str]) -> np.ndarray:
    """Load TensorFlow model and predict with feature alignment."""
    import tensorflow as tf

    model = tf.keras.models.load_model(model_path)

    # Determine expected input shape
    expected_n = None
    try:
        input_shape = model.input_shape
        if isinstance(input_shape, tuple) and len(input_shape) >= 2:
            expected_n = input_shape[-1]
    except Exception:
        pass

    scaler_path = Path(model_dir) / "scaler.pkl"
    scaler = None
    if scaler_path.exists():
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        if expected_n is None:
            if hasattr(scaler, 'n_features_in_'):
                expected_n = scaler.n_features_in_
            elif hasattr(scaler, 'mean_') and scaler.mean_ is not None:
                expected_n = len(scaler.mean_)

    # Feature alignment
    X = df[features].values.astype(np.float32)
    if expected_n and len(features) != expected_n:
        if len(features) < expected_n:
            pad = np.zeros((len(df), expected_n - len(features)), dtype=np.float32)
            X = np.hstack([X, pad])
            logger.warning(f"  tensorflow: padded {expected_n - len(features)} missing features")
        else:
            X = X[:, :expected_n]
            logger.warning(f"  tensorflow: trimmed {len(features) - expected_n} extra features")

    if scaler is not None:
        X = scaler.transform(X)

    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)

    probs = model.predict(X, verbose=0)
    if probs.ndim == 2 and probs.shape[1] == 1:
        probs = probs.ravel()
    elif probs.ndim == 2 and probs.shape[1] == 2:
        probs = probs[:, 1]

    return np.clip(probs, 0.001, 0.999)'''

# ============================================================================
# PATCH 5: Updated _predict_autogluon with feature alignment
# ============================================================================

OLD_PREDICT_AG = '''def _predict_autogluon(model_path: str, df: pd.DataFrame,
                       features: List[str]) -> np.ndarray:
    """Load AutoGluon predictor and predict."""
    from autogluon.tabular import TabularPredictor

    predictor = TabularPredictor.load(model_path)
    probs = predictor.predict_proba(df[features])

    if isinstance(probs, pd.DataFrame) and probs.shape[1] == 2:
        return np.clip(probs.iloc[:, 1].values, 0.001, 0.999)
    elif isinstance(probs, pd.Series):
        return np.clip(probs.values, 0.001, 0.999)
    else:
        return np.clip(probs.values.ravel(), 0.001, 0.999)'''

NEW_PREDICT_AG = '''def _predict_autogluon(model_path: str, df: pd.DataFrame,
                       features: List[str]) -> np.ndarray:
    """Load AutoGluon predictor and predict with feature alignment."""
    from autogluon.tabular import TabularPredictor

    predictor = TabularPredictor.load(model_path)

    # AutoGluon knows its own features - align by adding missing cols with 0
    try:
        expected_features = predictor.feature_metadata_in.get_features()
    except Exception:
        expected_features = features

    pred_df = df[features].copy()

    # Add missing columns that AutoGluon expects
    missing = [f for f in expected_features if f not in pred_df.columns]
    if missing:
        logger.warning(f"  autogluon: adding {len(missing)} missing features as 0: {missing[:5]}...")
        for col in missing:
            pred_df[col] = 0.0

    probs = predictor.predict_proba(pred_df)

    if isinstance(probs, pd.DataFrame) and probs.shape[1] == 2:
        return np.clip(probs.iloc[:, 1].values, 0.001, 0.999)
    elif isinstance(probs, pd.Series):
        return np.clip(probs.values, 0.001, 0.999)
    else:
        return np.clip(probs.values.ravel(), 0.001, 0.999)'''

# ============================================================================
# PATCH 6: NEW _predict_h2o with categorical column fix (.asfactor())
# ============================================================================

NEW_PREDICT_H2O = '''def _predict_h2o(model_path: str, model_dir: str,
                 df: pd.DataFrame, features: List[str]) -> np.ndarray:
    """Load H2O model and predict with categorical column fix."""
    import h2o
    h2o.init(nthreads=-1, max_mem_size="4G", verbose=False)

    # Check for MOJO first
    mojo_dir = Path(model_dir) / "mojo"
    if mojo_dir.exists() and mojo_dir.is_dir():
        mojo_files = list(mojo_dir.glob("*.zip"))
        if mojo_files:
            model = h2o.import_mojo(str(mojo_files[0]))
        else:
            model = h2o.load_model(model_path)
    else:
        model = h2o.load_model(model_path)

    # Only use features that exist in the dataframe
    available_features = [f for f in features if f in df.columns]
    h2o_df = h2o.H2OFrame(df[available_features])

    # FIX: Convert columns to categorical if the H2O model expects them as such.
    # During training, H2O auto-converts low-cardinality numeric columns
    # (like consensus_spread with few unique values) to categorical.
    # At prediction time, we must match the training column types exactly.
    try:
        if hasattr(model, '_model_json'):
            model_output = model._model_json.get('output', {})
            domains = model_output.get('domains', None)
            names = model_output.get('names', None)
            if domains and names:
                converted = []
                for col_name, domain in zip(names, domains):
                    if domain is not None and col_name in h2o_df.columns:
                        try:
                            h2o_df[col_name] = h2o_df[col_name].asfactor()
                            converted.append(col_name)
                        except Exception:
                            pass
                if converted:
                    logger.info(f"  H2O: converted {len(converted)} cols to categorical: {converted[:5]}...")
    except Exception as e:
        logger.debug(f"Could not auto-detect categorical columns: {e}")

    preds = model.predict(h2o_df)
    probs = preds.as_data_frame()

    # H2O typically returns columns: predict, p0, p1
    if 'p1' in probs.columns:
        result = probs['p1'].values
    elif probs.shape[1] >= 3:
        result = probs.iloc[:, 2].values
    else:
        result = probs.iloc[:, 0].values

    return np.clip(result.astype(float), 0.001, 0.999)'''

# ============================================================================
# PATCH 7: Stratified validation split fallback
# ============================================================================

OLD_VAL_SPLIT = """    # Take last 20% as validation
    split_idx = int(len(df) * 0.8)
    val_df = df.iloc[split_idx:].copy().reset_index(drop=True)"""

NEW_VAL_SPLIT = """    # Take last 20% as validation (with stratified fallback)
    split_idx = int(len(df) * 0.8)
    val_df = df.iloc[split_idx:].copy().reset_index(drop=True)

    # FIX: If validation has only one class, AUC computation will fail.
    # Fall back to stratified split to ensure both classes are present.
    if target_col in val_df.columns and val_df[target_col].nunique() < 2:
        logger.warning(
            f"  ⚠️ Chronological split produced single-class validation "
            f"(only {val_df[target_col].unique()[0]}). Switching to stratified split."
        )
        try:
            from sklearn.model_selection import train_test_split
            _, val_df = train_test_split(
                df, test_size=0.2, random_state=42,
                stratify=df[target_col]
            )
            val_df = val_df.reset_index(drop=True)
            logger.info(
                f"  Stratified split: {len(val_df)} samples, "
                f"classes={val_df[target_col].value_counts().to_dict()}"
            )
        except Exception as e:
            logger.warning(f"  Stratified split also failed: {e}")"""


# ============================================================================
# APPLY ALL PATCHES (v3 + v4 combined)
# ============================================================================

def apply_patches():
    if not EVAL_SCRIPT.exists():
        print(f"ERROR: {EVAL_SCRIPT} not found")
        sys.exit(1)

    # Restore from original backup if it exists (ensures clean base)
    bak = EVAL_SCRIPT.with_suffix('.py.bak')
    if bak.exists():
        print(f"  Restoring original from {bak} before patching...")
        content = bak.read_text()
        EVAL_SCRIPT.write_text(content)
    else:
        content = EVAL_SCRIPT.read_text()

    original = content
    changes = []

    # ══════════════════════════════════════════════════════════════════
    # v4 FIX 1: Suppress PerformanceWarnings
    # ══════════════════════════════════════════════════════════════════
    if 'import warnings' not in content:
        old_imports = "import numpy as np\nimport pandas as pd"
        new_imports = """import warnings
import numpy as np
import pandas as pd

# Suppress DataFrame fragmentation warnings during feature engineering
# (we call df.copy() at the end to defragment)
warnings.filterwarnings('ignore', message='.*DataFrame is highly fragmented.*')"""
        if old_imports in content:
            content = content.replace(old_imports, new_imports)
            changes.append("[v4] Added warnings suppression for PerformanceWarning")
        else:
            print("  WARNING: Could not find import block for warnings insertion")
    else:
        print("  SKIP: warnings already imported")

    # ══════════════════════════════════════════════════════════════════
    # v4 FIX 2: Add tiered threshold constants
    # ══════════════════════════════════════════════════════════════════
    old_thresholds = """# Minimum acceptable thresholds
MIN_ACCURACY = 0.52
MIN_AUC = 0.52
MAX_CALIBRATION_ERROR = 0.15"""

    new_thresholds = """# Minimum acceptable thresholds
MIN_ACCURACY = 0.52
MIN_AUC = 0.52
MAX_CALIBRATION_ERROR = 0.15

# Tiered pass criteria (more nuanced than binary pass/fail)
# PRODUCTION: ready to bet with real money
# SIGNAL: model has real discriminative power, needs calibration work
# LEAKAGE_SUSPECT: high accuracy from class imbalance, not model skill
TIER_SIGNAL_AUC = 0.55        # AUC alone proves ranking ability
TIER_LEAKAGE_CLASS_PCT = 0.88  # If >88% predictions are one class → suspect"""

    if old_thresholds in content:
        content = content.replace(old_thresholds, new_thresholds)
        changes.append("[v4] Added tiered threshold constants (SIGNAL/LEAKAGE)")
    else:
        print("  SKIP: thresholds already modified")

    # ══════════════════════════════════════════════════════════════════
    # v4 FIX 3: Add tier/leakage fields to ModelScore dataclass
    # ══════════════════════════════════════════════════════════════════
    old_dataclass_end = """    # Ranking
    composite_score: float = 0.0
    rank: int = 0
    passes_threshold: bool = False"""

    new_dataclass_end = """    # Ranking
    composite_score: float = 0.0
    rank: int = 0
    passes_threshold: bool = False
    tier: str = ""          # "PRODUCTION", "SIGNAL", "WEAK", "LEAKAGE_SUSPECT"
    leakage_suspect: bool = False
    dominant_class_pct: float = 0.0  # % of predictions in majority class"""

    if old_dataclass_end in content and 'tier: str' not in content:
        content = content.replace(old_dataclass_end, new_dataclass_end)
        changes.append("[v4] Added tier/leakage fields to ModelScore")
    else:
        print("  SKIP: ModelScore already has tier fields")

    # ══════════════════════════════════════════════════════════════════
    # v4 FIX 4: Fix composite_score - realistic ROI range + leakage penalty
    # ══════════════════════════════════════════════════════════════════
    old_composite = '''def compute_composite_score(metrics: Dict[str, float]) -> float:
    """
    Composite score for ranking models.
    Weighted combination: AUC(30%) + Accuracy(25%) + ROI(25%) + (1-ECE)(10%) + (1-Brier)(10%)
    """
    auc = metrics.get("auc_roc", 0.5)
    acc = metrics.get("accuracy", 0.5)
    roi = max(metrics.get("simulated_roi", -1.0), -1.0)  # Clamp
    ece = min(metrics.get("ece", 1.0), 1.0)
    brier = min(metrics.get("brier_score", 1.0), 1.0)

    # Normalize ROI to 0-1 range (assume -50% to +50%)
    roi_norm = (roi + 0.5) / 1.0
    roi_norm = max(0, min(1, roi_norm))

    score = (
        0.30 * auc +
        0.25 * acc +
        0.25 * roi_norm +
        0.10 * (1.0 - ece) +
        0.10 * (1.0 - brier)
    )

    return round(score, 6)'''

    new_composite = '''def compute_composite_score(metrics: Dict[str, float],
                            leakage_suspect: bool = False) -> float:
    """
    Composite score for ranking models.
    AUC(35%) + Accuracy(20%) + ROI(20%) + (1-ECE)(15%) + (1-Brier)(10%)
    
    Changes from v3:
    - AUC weighted higher (best single metric for model quality)
    - ECE weighted higher (calibration matters for betting)
    - ROI capped at realistic +30% (prevents tennis leakage inflation)
    - Leakage penalty halves score for suspect models
    """
    auc = metrics.get("auc_roc", 0.5)
    acc = metrics.get("accuracy", 0.5)
    roi = max(metrics.get("simulated_roi", -1.0), -1.0)
    ece = min(metrics.get("ece", 1.0), 1.0)
    brier = min(metrics.get("brier_score", 1.0), 1.0)

    # Normalize ROI to 0-1 range with REALISTIC bounds (-20% to +30%)
    # Any ROI > 30% is almost certainly leakage/overfitting
    roi_capped = max(-0.20, min(0.30, roi))
    roi_norm = (roi_capped + 0.20) / 0.50
    roi_norm = max(0, min(1, roi_norm))

    score = (
        0.35 * auc +
        0.20 * acc +
        0.20 * roi_norm +
        0.15 * (1.0 - ece) +
        0.10 * (1.0 - brier)
    )

    # Leakage penalty: models that predict one class >88% of the time
    # get a heavy penalty (likely exploiting class imbalance, not skill)
    if leakage_suspect:
        score *= 0.50

    return round(score, 6)'''

    if old_composite in content:
        content = content.replace(old_composite, new_composite)
        changes.append("[v4] Fixed composite_score: realistic ROI cap + AUC-weighted + leakage penalty")
    else:
        print("  SKIP: composite_score already modified")

    # ══════════════════════════════════════════════════════════════════
    # v4 FIX 5: Replace threshold check with tiered system + leakage detection
    # ══════════════════════════════════════════════════════════════════
    old_threshold_block = """                # Composite score
                score.composite_score = compute_composite_score(metrics)

                # Threshold check
                score.passes_threshold = (
                    score.accuracy >= MIN_ACCURACY and
                    score.auc_roc >= MIN_AUC and
                    score.ece <= MAX_CALIBRATION_ERROR
                )

                if verbose:
                    status = "✅" if score.passes_threshold else "❌"
                    console.print(
                        f"  {status} {framework}/{sport}/{bet_type}: "
                        f"acc={score.accuracy:.3f} auc={score.auc_roc:.3f} "
                        f"roi={score.simulated_roi:+.3f} ece={score.ece:.3f}"
                    )"""

    new_threshold_block = """                # Leakage detection: if model predicts one class >88% of the time,
                # "accuracy" comes from class imbalance not model skill
                pred_classes = (y_pred_proba > 0.5).astype(int)
                majority_pct = max(pred_classes.mean(), 1 - pred_classes.mean())
                score.dominant_class_pct = majority_pct
                score.leakage_suspect = majority_pct > TIER_LEAKAGE_CLASS_PCT

                # Composite score (with leakage penalty)
                score.composite_score = compute_composite_score(
                    metrics, leakage_suspect=score.leakage_suspect
                )

                # Tiered threshold check
                score.passes_threshold = (
                    score.accuracy >= MIN_ACCURACY and
                    score.auc_roc >= MIN_AUC and
                    score.ece <= MAX_CALIBRATION_ERROR and
                    not score.leakage_suspect
                )

                # Assign tier
                if score.leakage_suspect:
                    score.tier = "LEAK?"
                elif score.passes_threshold:
                    score.tier = "PROD"
                elif score.auc_roc >= TIER_SIGNAL_AUC:
                    score.tier = "SIGNAL"
                else:
                    score.tier = "WEAK"

                if verbose:
                    tier_colors = {"PROD": "green", "SIGNAL": "yellow", "LEAK?": "red", "WEAK": "dim"}
                    color = tier_colors.get(score.tier, "white")
                    icon = {"PROD": "✅", "SIGNAL": "📊", "LEAK?": "⚠️", "WEAK": "❌"}.get(score.tier, "❌")
                    console.print(
                        f"  {icon} [{color}]{framework}/{sport}/{bet_type}[/{color}]: "
                        f"acc={score.accuracy:.3f} auc={score.auc_roc:.3f} "
                        f"roi={score.simulated_roi:+.3f} ece={score.ece:.3f} "
                        f"[{color}][{score.tier}][/{color}]"
                    )"""

    if old_threshold_block in content:
        content = content.replace(old_threshold_block, new_threshold_block)
        changes.append("[v4] Added tiered pass system (PROD/SIGNAL/LEAK?/WEAK) + leakage detection")
    else:
        print("  SKIP: threshold block already modified")

    # ══════════════════════════════════════════════════════════════════
    # v4 FIX 6: Update scorecard display to show tiers
    # ══════════════════════════════════════════════════════════════════
    old_pass_str = '        pass_str = "[green]✓[/green]" if s.passes_threshold else "[red]✗[/red]"'
    new_pass_str = '''        tier_colors = {"PROD": "green", "SIGNAL": "yellow", "LEAK?": "red", "WEAK": "dim"}
        color = tier_colors.get(s.tier, "white")
        pass_str = f"[{color}]{s.tier}[/{color}]"'''

    if old_pass_str in content:
        content = content.replace(old_pass_str, new_pass_str)
        changes.append("[v4] Updated scorecard display with tier labels")
    else:
        print("  SKIP: pass_str display already modified")

    # Also update the column header
    old_col = '    table.add_column("Pass?", style="white")'
    new_col = '    table.add_column("Tier", style="white")'
    if old_col in content:
        content = content.replace(old_col, new_col)

    # ══════════════════════════════════════════════════════════════════
    # v4 FIX 7: Update summary to show tiered breakdown
    # ══════════════════════════════════════════════════════════════════
    old_summary = '''def _print_summary(scores: List[ModelScore]):
    """Print summary statistics."""
    loaded = [s for s in scores if s.load_success]
    passed = [s for s in loaded if s.passes_threshold]
    failed = [s for s in loaded if not s.passes_threshold]
    errors = [s for s in scores if not s.load_success]

    console.print(Panel(
        f"[bold]Total Models:[/bold] {len(scores)}\\n"
        f"[green]Loaded Successfully:[/green] {len(loaded)}\\n"
        f"[green]Passed Thresholds:[/green] {len(passed)} (acc≥{MIN_ACCURACY}, auc≥{MIN_AUC}, ece≤{MAX_CALIBRATION_ERROR})\\n"
        f"[red]Below Threshold:[/red] {len(failed)}\\n"
        f"[red]Load Errors:[/red] {len(errors)}",
        title="Summary"
    ))

    # Best per sport/bet_type
    if passed:
        console.print("\\n[cyan]Best Model per Sport/Bet-Type:[/cyan]")
        best_table = Table()
        best_table.add_column("Sport", style="cyan")
        best_table.add_column("Bet Type", style="blue")
        best_table.add_column("Framework", style="magenta")
        best_table.add_column("Accuracy", style="yellow")
        best_table.add_column("AUC", style="yellow")
        best_table.add_column("ROI", style="green")
        best_table.add_column("Composite", style="bold white")

        seen = set()
        for s in passed:
            key = (s.sport, s.bet_type)
            if key in seen:
                continue
            seen.add(key)
            best_table.add_row(
                s.sport, s.bet_type, s.framework,
                f"{s.accuracy:.3f}", f"{s.auc_roc:.3f}",
                f"{s.simulated_roi*100:+.1f}%", f"{s.composite_score:.4f}",
            )'''

    new_summary = '''def _print_summary(scores: List[ModelScore]):
    """Print summary statistics with tiered breakdown."""
    loaded = [s for s in scores if s.load_success]
    errors = [s for s in scores if not s.load_success]

    # Tiered counts
    prod = [s for s in loaded if s.tier == "PROD"]
    signal = [s for s in loaded if s.tier == "SIGNAL"]
    leakage = [s for s in loaded if s.tier == "LEAK?"]
    weak = [s for s in loaded if s.tier == "WEAK"]

    console.print(Panel(
        f"[bold]Total Models:[/bold] {len(scores)}\\n"
        f"[green]Loaded Successfully:[/green] {len(loaded)}\\n"
        f"[red]Load Errors:[/red] {len(errors)}\\n"
        f"\\n"
        f"[bold]Model Tiers:[/bold]\\n"
        f"  [green]🟢 PRODUCTION:[/green]  {len(prod):>3}  (acc≥{MIN_ACCURACY}, auc≥{MIN_AUC}, ece≤{MAX_CALIBRATION_ERROR})\\n"
        f"  [yellow]🟡 SIGNAL:[/yellow]      {len(signal):>3}  (auc≥{TIER_SIGNAL_AUC}, needs calibration/threshold work)\\n"
        f"  [red]🔴 LEAK?:[/red]       {len(leakage):>3}  (>88% predictions in one class)\\n"
        f"  [dim]⚪ WEAK:[/dim]        {len(weak):>3}  (no discriminative power)",
        title="Summary"
    ))

    # Show SIGNAL models (these have edge but need work)
    if signal:
        console.print("\\n[yellow]📊 SIGNAL Models (have edge, need calibration):[/yellow]")
        sig_table = Table()
        sig_table.add_column("Sport", style="cyan")
        sig_table.add_column("Bet Type", style="blue")
        sig_table.add_column("Framework", style="magenta")
        sig_table.add_column("AUC", style="yellow")
        sig_table.add_column("Acc", style="yellow")
        sig_table.add_column("ECE", style="red")
        sig_table.add_column("ROI", style="green")
        sig_table.add_column("Issue", style="dim")

        for s in sorted(signal, key=lambda x: x.auc_roc, reverse=True):
            issues = []
            if s.accuracy < MIN_ACCURACY:
                issues.append(f"acc<{MIN_ACCURACY}")
            if s.ece > MAX_CALIBRATION_ERROR:
                issues.append(f"ece>{MAX_CALIBRATION_ERROR}")
            sig_table.add_row(
                s.sport, s.bet_type, s.framework,
                f"{s.auc_roc:.3f}", f"{s.accuracy:.3f}", f"{s.ece:.3f}",
                f"{s.simulated_roi*100:+.1f}%",
                ", ".join(issues) if issues else "threshold border",
            )
        console.print(sig_table)

    # Best PRODUCTION models per sport/bet_type
    if prod:
        console.print("\\n[green]✅ PRODUCTION Models (ready for deployment):[/green]")
        best_table = Table()
        best_table.add_column("Sport", style="cyan")
        best_table.add_column("Bet Type", style="blue")
        best_table.add_column("Framework", style="magenta")
        best_table.add_column("Accuracy", style="yellow")
        best_table.add_column("AUC", style="yellow")
        best_table.add_column("ROI", style="green")
        best_table.add_column("Composite", style="bold white")

        seen = set()
        for s in prod:
            key = (s.sport, s.bet_type)
            if key in seen:
                continue
            seen.add(key)
            best_table.add_row(
                s.sport, s.bet_type, s.framework,
                f"{s.accuracy:.3f}", f"{s.auc_roc:.3f}",
                f"{s.simulated_roi*100:+.1f}%", f"{s.composite_score:.4f}",
            )'''

    if old_summary in content:
        content = content.replace(old_summary, new_summary)
        changes.append("[v4] Updated summary with tiered breakdown + SIGNAL model details")
    else:
        print("  SKIP: summary already modified")

    # ══════════════════════════════════════════════════════════════════
    # v4 FIX 8: Update CSV export to include new fields
    # ══════════════════════════════════════════════════════════════════
    old_csv_fields = '''            "composite_score": round(s.composite_score, 6),
            "passes_threshold": s.passes_threshold,'''
    new_csv_fields = '''            "composite_score": round(s.composite_score, 6),
            "passes_threshold": s.passes_threshold,
            "tier": s.tier,
            "leakage_suspect": s.leakage_suspect,
            "dominant_class_pct": round(s.dominant_class_pct, 3),'''
    if old_csv_fields in content:
        content = content.replace(old_csv_fields, new_csv_fields)
        changes.append("[v4] Added tier/leakage fields to CSV export")
    else:
        print("  SKIP: CSV fields already modified")

    # ══════════════════════════════════════════════════════════════════
    # v3 PATCH 1: Add engineer_features function
    # ══════════════════════════════════════════════════════════════════
    if 'def engineer_features(' not in content:
        marker = "# ============================================================================\n# VALIDATION DATA LOADING"
        if marker in content:
            content = content.replace(
                marker,
                "# ============================================================================\n"
                "# FEATURE ENGINEERING (replicated from training_service.py)\n"
                "# ============================================================================\n"
                + ENGINEER_FEATURES_FUNC + "\n"
                + marker
            )
            changes.append("[v3] Added engineer_features() function (30+ derived features + df.copy())")
        else:
            print("WARNING: Could not find VALIDATION DATA LOADING marker")
    else:
        print("  SKIP: engineer_features() already exists")

    # ══════════════════════════════════════════════════════════════════
    # v3 PATCH 2: Call engineer_features in load_validation_data
    # ══════════════════════════════════════════════════════════════════
    call_marker = "    # Identify feature columns (exclude meta/target/identifier columns)"
    if 'def load_validation_data' in content:
        load_func_section = content.split('def load_validation_data')[1].split('\ndef ')[0]
        if 'engineer_features(' not in load_func_section:
            if call_marker in content:
                content = content.replace(
                    call_marker,
                    "    # ── Apply feature engineering (create derived features matching training) ──\n"
                    "    df = engineer_features(df, sport)\n\n"
                    + call_marker
                )
                changes.append("[v3] Added engineer_features() call in load_validation_data()")
        else:
            print("  SKIP: engineer_features() call already in load_validation_data()")

    # ══════════════════════════════════════════════════════════════════
    # v3 PATCH 3: Replace load_and_predict
    # ══════════════════════════════════════════════════════════════════
    if OLD_LOAD_AND_PREDICT in content:
        content = content.replace(OLD_LOAD_AND_PREDICT, NEW_LOAD_AND_PREDICT)
        changes.append("[v3] Updated load_and_predict() with better error logging")
    else:
        print("  SKIP: load_and_predict already modified or different format")

    # ══════════════════════════════════════════════════════════════════
    # v3 PATCH 4: Replace _predict_sklearn
    # ══════════════════════════════════════════════════════════════════
    if OLD_PREDICT_SKLEARN in content:
        content = content.replace(OLD_PREDICT_SKLEARN, NEW_PREDICT_SKLEARN)
        changes.append("[v3] Updated _predict_sklearn() with feature alignment")
    else:
        print("  SKIP: _predict_sklearn already modified or different format")

    # ══════════════════════════════════════════════════════════════════
    # v3 PATCH 5: Replace _predict_tensorflow
    # ══════════════════════════════════════════════════════════════════
    if OLD_PREDICT_TF in content:
        content = content.replace(OLD_PREDICT_TF, NEW_PREDICT_TF)
        changes.append("[v3] Updated _predict_tensorflow() with feature alignment")
    else:
        print("  SKIP: _predict_tensorflow already modified or different format")

    # ══════════════════════════════════════════════════════════════════
    # v3 PATCH 6: Replace _predict_autogluon
    # ══════════════════════════════════════════════════════════════════
    if OLD_PREDICT_AG in content:
        content = content.replace(OLD_PREDICT_AG, NEW_PREDICT_AG)
        changes.append("[v3] Updated _predict_autogluon() with feature alignment")
    else:
        print("  SKIP: _predict_autogluon already modified or different format")

    # ══════════════════════════════════════════════════════════════════
    # v3 PATCH 7: Replace _predict_h2o with categorical fix
    # ══════════════════════════════════════════════════════════════════
    h2o_start = "def _predict_h2o("
    h2o_next = "\ndef _predict_autogluon("
    if h2o_start in content and h2o_next in content and 'asfactor' not in content:
        start_idx = content.index(h2o_start)
        end_idx = content.index(h2o_next, start_idx)
        content = content[:start_idx] + NEW_PREDICT_H2O + "\n\n" + content[end_idx:]
        changes.append("[v3] Updated _predict_h2o() with categorical .asfactor() fix")
    elif 'asfactor' in content:
        print("  SKIP: _predict_h2o already has asfactor fix")
    else:
        print("  WARNING: Could not locate _predict_h2o boundaries")

    # ══════════════════════════════════════════════════════════════════
    # v3 PATCH 8: Stratified validation split fallback
    # ══════════════════════════════════════════════════════════════════
    if OLD_VAL_SPLIT in content:
        content = content.replace(OLD_VAL_SPLIT, NEW_VAL_SPLIT)
        changes.append("[v3] Added stratified fallback for single-class validation splits")
    else:
        print("  SKIP: Validation split already modified or different format")

    # ══════════════════════════════════════════════════════════════════
    # v3 PATCH 9: Fix exclude_cols to NOT exclude spread_close and total_close
    # ══════════════════════════════════════════════════════════════════
    old_exclude = """    exclude_cols = {
        'game_id', 'date', 'scheduled_at', 'home_team', 'away_team',
        'home_team_id', 'away_team_id', 'season', 'season_type',
        'home_score', 'away_score', 'home_win', 'spread_result', 'over_result',
        'spread_close', 'total_close', 'home_ml', 'away_ml',
        'home_odds', 'away_odds', 'over_odds', 'under_odds',
        'spread_home_close', 'spread_away_close',
        'Unnamed: 0', 'index',
    }"""
    new_exclude = """    exclude_cols = {
        'game_id', 'date', 'scheduled_at', 'home_team', 'away_team',
        'home_team_id', 'away_team_id', 'season', 'season_type',
        'home_score', 'away_score', 'home_win', 'spread_result', 'over_result',
        'home_odds', 'away_odds', 'over_odds', 'under_odds',
        'Unnamed: 0', 'index',
    }
    # NOTE: spread_close, total_close etc are KEPT as features (used for
    # value calculations like spread_value, total_value etc)"""
    if old_exclude in content:
        content = content.replace(old_exclude, new_exclude)
        changes.append("[v3] Fixed exclude_cols: kept spread_close/total_close as features")
    else:
        print("  SKIP: exclude_cols already modified")

    # ── WRITE ──
    if changes:
        # Backup original (only if not already backed up)
        backup = EVAL_SCRIPT.with_suffix('.py.bak')
        if not backup.exists():
            backup.write_text(original)
            print(f"\n  Backup saved: {backup}")
        else:
            print(f"\n  Backup already exists: {backup}")

        EVAL_SCRIPT.write_text(content)
        print(f"\n{'='*60}")
        print(f"  PATCH APPLIED SUCCESSFULLY ({len(changes)} changes)")
        print(f"{'='*60}")
        for i, c in enumerate(changes, 1):
            print(f"  {i}. {c}")
        print(f"\n  Re-run evaluation:")
        print(f"    python scripts/evaluate_models.py --verbose")
    else:
        print("\n  No changes needed - all patches already applied")


if __name__ == "__main__":
    print("=" * 60)
    print("  ROYALEY Evaluation Patch (COMBINED v3 + v4)")
    print("  Fixes: features + H2O + splits + alignment")
    print("  + warnings + tiers + leakage + scoring")
    print("=" * 60)
    apply_patches()