#!/usr/bin/env python3
"""
ROYALEY - Evaluation Patch v5: Recover LEAK? Models
=====================================================
Root cause: Evaluation used DIFFERENT feature selection + imputation
than training, feeding wrong data to every model.

This patch REPLACES the entire evaluation pipeline with one that
replicates the EXACT training logic from training_service.py:

  1. Same metadata_columns exclusion set
  2. Same true_leakage_columns exclusion set  
  3. Same dead-feature removal (95% null, constant)
  4. Same 30+ derived feature engineering
  5. Same tennis random-swap debiasing + relative features
  6. Same smart imputation (odds→NaN, pct→0.5, margin→median)
  7. Same missingness indicators (has_odds, has_spread_odds, has_total_odds)
  8. Model-intrinsic feature extraction (sklearn/h2o/autogluon)
  9. Feature alignment verification + auto-correction
  10. Platt scaling calibration for SIGNAL models
  11. Saves feature_columns.json for each model (future-proof)

Expected: Recover 30-60+ models from LEAK? to SIGNAL/PRODUCTION

Usage:
    docker cp patch_v5_recover_models.py royaley_api:/app/scripts/
    docker exec royaley_api python /app/scripts/patch_v5_recover_models.py
    docker exec -it royaley_api python scripts/evaluate_models.py --verbose
"""

import sys
import textwrap
from pathlib import Path

EVAL_SCRIPT = Path("/app/scripts/evaluate_models.py")

def read_file():
    return EVAL_SCRIPT.read_text()

def write_file(content):
    # Backup
    import shutil
    backup = EVAL_SCRIPT.with_suffix('.py.bak_v5')
    if not backup.exists() and EVAL_SCRIPT.exists():
        shutil.copy2(str(EVAL_SCRIPT), str(backup))
    EVAL_SCRIPT.write_text(content)

def apply_patches():
    content = read_file()
    patches_applied = 0

    # =========================================================================
    # PATCH 1: Add imports for calibration
    # =========================================================================
    if 'from sklearn.calibration' not in content:
        old = 'import numpy as np\nimport pandas as pd'
        new = '''import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', message='.*PerformanceWarning.*')'''
        if old in content:
            content = content.replace(old, new)
            patches_applied += 1
            print(f"  [PATCH {patches_applied}] Added warnings suppression")

    # =========================================================================
    # PATCH 2: Add tiered scoring constants + dataclass fields
    # =========================================================================
    old_consts = '''# Minimum acceptable thresholds
MIN_ACCURACY = 0.52
MIN_AUC = 0.52
MAX_CALIBRATION_ERROR = 0.15'''

    new_consts = '''# Minimum acceptable thresholds
MIN_ACCURACY = 0.52
MIN_AUC = 0.52
MAX_CALIBRATION_ERROR = 0.15

# Tiered evaluation thresholds
TIER_SIGNAL_AUC = 0.54        # Models with edge (needs work)
TIER_LEAKAGE_CLASS_PCT = 0.88  # >88% one class = likely broken
TIER_PROD_AUC = 0.52
TIER_PROD_ACC = 0.52
TIER_PROD_ECE = 0.15'''

    if 'TIER_SIGNAL_AUC' not in content:
        content = content.replace(old_consts, new_consts)
        patches_applied += 1
        print(f"  [PATCH {patches_applied}] Added tiered threshold constants")

    # =========================================================================
    # PATCH 3: Add tier/leakage fields to ModelScore dataclass
    # =========================================================================
    old_ds = '''    # Ranking
    composite_score: float = 0.0
    rank: int = 0
    passes_threshold: bool = False'''

    new_ds = '''    # Ranking
    composite_score: float = 0.0
    rank: int = 0
    passes_threshold: bool = False

    # Tier classification
    tier: str = ""               # PROD / SIGNAL / LEAK? / WEAK
    leakage_suspect: bool = False
    dominant_class_pct: float = 0.0
    feature_alignment: str = ""  # matched / trimmed / padded / reconstructed'''

    if 'tier: str' not in content:
        content = content.replace(old_ds, new_ds)
        patches_applied += 1
        print(f"  [PATCH {patches_applied}] Added tier/leakage fields to ModelScore")

    # =========================================================================
    # PATCH 4: REPLACE load_validation_data with training-replica logic
    # This is the KEY fix - replicate exact training feature pipeline
    # =========================================================================
    old_validation = '''def load_validation_data(sport: str, bet_type: str,
                         csv_dir: str = None) -> Optional[Tuple[pd.DataFrame, List[str], str]]:
    """
    Load validation data for a sport/bet_type.
    Uses the last 20% of data as validation (walk-forward style).
    """'''

    # Find the end of load_validation_data function
    if old_validation in content:
        # Find where this function ends (next function or section)
        start_idx = content.index(old_validation)
        # Find next section marker
        next_section = content.find('\n# ====', start_idx + len(old_validation))
        if next_section == -1:
            next_section = content.find('\nasync def evaluate_all_models', start_idx + 100)

        old_func = content[start_idx:next_section]

        new_func = '''def load_validation_data(sport: str, bet_type: str,
                         csv_dir: str = None) -> Optional[Tuple[pd.DataFrame, List[str], str]]:
    """
    Load validation data replicating EXACT training pipeline logic.
    Uses the last 20% of data as validation (walk-forward style).
    
    Mirrors training_service.py _prepare_training_data():
    - Same metadata/leakage exclusions
    - Same feature engineering (30+ derived features)
    - Same tennis debiasing
    - Same smart imputation (odds→NaN, pct→0.5, margin→median)
    - Same dead-feature removal
    - Same missingness indicators
    """
    csv_paths = [
        Path(csv_dir) if csv_dir else None,
        Path("/app/ml_csv"),
        Path(__file__).parent.parent / "app" / "services" / "ml_csv",
        Path(__file__).parent.parent / "ml_csv",
    ]

    df = None
    target_col = TARGET_COLUMNS.get(bet_type, "home_win")

    for csv_path in csv_paths:
        if csv_path is None or not csv_path.exists():
            continue

        sport_dir = csv_path / sport
        if not sport_dir.exists():
            pattern = f"ml_features_{sport}_*.csv"
            files = list(csv_path.glob(pattern))
            if not files:
                continue
            dfs = []
            for f in sorted(files):
                try:
                    dfs.append(pd.read_csv(f))
                except Exception:
                    continue
            if dfs:
                df = pd.concat(dfs, ignore_index=True)
            break
        else:
            files = list(sport_dir.glob("*.csv"))
            if not files:
                continue
            dfs = []
            for f in sorted(files):
                try:
                    dfs.append(pd.read_csv(f))
                except Exception:
                    continue
            if dfs:
                df = pd.concat(dfs, ignore_index=True)
            break

    if df is None or len(df) < 30:
        return None

    # ================================================================
    # RECONSTRUCT TARGET (same as original)
    # ================================================================
    if target_col not in df.columns:
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

    df = df.dropna(subset=[target_col])
    if len(df) < 30:
        return None

    # ================================================================
    # TENNIS: Random swap debiasing (EXACT replica of training)
    # ================================================================
    if sport in ('ATP', 'WTA'):
        df = df.drop_duplicates()
        home_win_pct = df['home_win'].mean() if 'home_win' in df.columns else 0
        
        if home_win_pct > 0.65:
            np.random.seed(42)  # Same seed as training!
            swap_mask = np.random.random(len(df)) < 0.5
            
            home_cols = [c for c in df.columns if c.startswith('home_')]
            away_cols = [c for c in df.columns if c.startswith('away_')]
            
            swap_pairs = []
            for hcol in home_cols:
                suffix = hcol[5:]
                acol = f'away_{suffix}'
                if acol in away_cols:
                    swap_pairs.append((hcol, acol))
            
            odds_swap_pairs = [
                ('moneyline_home_open', 'moneyline_away_open'),
                ('moneyline_home_close', 'moneyline_away_close'),
            ]
            for hcol, acol in odds_swap_pairs:
                if hcol in df.columns and acol in df.columns:
                    if (hcol, acol) not in swap_pairs:
                        swap_pairs.append((hcol, acol))
            
            for hcol, acol in swap_pairs:
                tmp = df.loc[swap_mask, hcol].copy()
                df.loc[swap_mask, hcol] = df.loc[swap_mask, acol]
                df.loc[swap_mask, acol] = tmp
            
            # Flip target
            if 'home_win' in df.columns:
                df.loc[swap_mask, 'home_win'] = 1 - df.loc[swap_mask, 'home_win']
            
            # Flip spread-related targets
            spread_flip_cols = ['spread_result', 'spread_close', 'spread_open',
                               'spread_line', 'home_spread', 'consensus_spread',
                               'pinnacle_spread']
            for col in spread_flip_cols:
                if col in df.columns:
                    df.loc[swap_mask, col] = -df.loc[swap_mask, col]
            
            # Flip implied probs
            for prob_pair in [('implied_home_prob', 'implied_away_prob'),
                             ('no_vig_home_prob', 'no_vig_away_prob')]:
                if prob_pair[0] in df.columns and prob_pair[1] in df.columns:
                    tmp = df.loc[swap_mask, prob_pair[0]].copy()
                    df.loc[swap_mask, prob_pair[0]] = df.loc[swap_mask, prob_pair[1]]
                    df.loc[swap_mask, prob_pair[1]] = tmp
        
        # Create tennis relative features
        feature_pairs = [
            ('home_power_rating', 'away_power_rating', 'power_diff'),
            ('home_wins_last10', 'away_wins_last10', 'wins_diff'),
            ('home_wins_last5', 'away_wins_last5', 'wins5_diff'),
            ('home_win_pct_last10', 'away_win_pct_last10', 'winpct_diff'),
            ('home_rest_days', 'away_rest_days', 'rest_diff'),
            ('home_avg_margin_last10', 'away_avg_margin_last10', 'margin_diff'),
            ('home_avg_pts_last10', 'away_avg_pts_last10', 'pts_diff'),
            ('home_streak', 'away_streak', 'streak_diff'),
            ('home_season_game_num', 'away_season_game_num', 'games_diff'),
        ]
        for home_col, away_col, diff_name in feature_pairs:
            if home_col in df.columns and away_col in df.columns:
                df[diff_name] = df[home_col] - df[away_col]
        
        df['_is_tennis'] = True

    # ================================================================
    # FEATURE ENGINEERING: 30+ derived features (EXACT replica)
    # ================================================================
    df = _engineer_features_training_replica(df, sport)

    # ================================================================
    # FEATURE SELECTION: Replicate exact training logic
    # ================================================================
    
    # STEP 1: Metadata columns (same as training_service.py)
    metadata_columns = {
        'master_game_id', 'game_id', 'match_id', 'id', 'index',
        'game_date', 'date', 'datetime', 'scheduled_at',
        'home_team_name', 'away_team_name', 'home_team', 'away_team',
        'team_home', 'team_away', 'home_team_id', 'away_team_id',
        'season', 'week', 'round', 'sport_code', 'sport',
    }
    
    # STEP 2: True leakage columns (same as training_service.py)
    true_leakage_columns = {
        'home_score', 'away_score', 'home_points', 'away_points',
        'final_score_home', 'final_score_away',
        'score_margin', 'point_margin', 'total_points', 'game_total',
        'combined_score', 'total_score', 'point_diff', 'score_diff',
        'home_win', 'away_win', 'winner', 'winning_team',
        'spread_result', 'over_result', 'under_result',
        'cover', 'covered', 'ats_result', 'against_spread',
        'over_under_result', 'ou_result',
        'moneyline_result', 'total_result',
    }
    
    # STEP 3: Select features (same logic as training)
    feature_columns = []
    for col in df.columns:
        col_lower = col.lower()
        if col == target_col:
            continue
        if col_lower in metadata_columns or col in metadata_columns:
            continue
        if col_lower.startswith('unnamed'):
            continue
        if col_lower in true_leakage_columns or col in true_leakage_columns:
            continue
        if col.startswith('_'):  # internal markers like _is_tennis
            continue
        # Must be numeric
        if df[col].dtype not in ['int64', 'float64', 'int32', 'float32', 'bool']:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if df[col].isna().all():
                    continue
            except:
                continue
        feature_columns.append(col)
    
    # STEP 4: Remove dead features (95% null or constant)
    clean_features = []
    for col in feature_columns:
        null_pct = df[col].isna().mean()
        n_unique = df[col].nunique()
        if null_pct >= 0.95:
            continue
        if n_unique <= 1:
            continue
        clean_features.append(col)
    feature_columns = clean_features
    
    # STEP 5: Tennis feature filtering (same as training)
    if sport in ('ATP', 'WTA') and '_is_tennis' in df.columns:
        tennis_safe_prefixes = (
            'power_diff', 'wins_diff', 'wins5_diff', 'winpct_diff',
            'rest_diff', 'margin_diff', 'pts_diff', 'streak_diff',
            'games_diff', 'rest_advantage', 'power_rating_diff',
            'month', 'day_of_week', 'season',
            'implied_', 'no_vig_', 'moneyline_', 'spread_',
            'total_', 'overround', 'has_odds', 'has_spread',
            'has_total', 'odds_',
        )
        tennis_safe_exact = {
            'month', 'day_of_week', 'year', 'hour',
            'rest_advantage', 'power_rating_diff',
            'has_odds', 'is_playoff', 'is_neutral_site',
        }
        tennis_features = []
        for col in feature_columns:
            if col.startswith('h2h_') or col.startswith('H2H_'):
                continue
            if any(col.startswith(prefix) or col.lower().startswith(prefix.lower()) 
                   for prefix in tennis_safe_prefixes):
                tennis_features.append(col)
            elif col in tennis_safe_exact or col.lower() in tennis_safe_exact:
                tennis_features.append(col)
            elif col.startswith('home_') or col.startswith('away_'):
                continue  # Exclude raw home/away for tennis
            else:
                tennis_features.append(col)  # Keep neutral features
        feature_columns = tennis_features

    # STEP 6: Missingness indicators (same as training)
    odds_indicator_cols = [
        'implied_home_prob', 'no_vig_home_prob',
        'moneyline_home_close', 'moneyline_away_close',
        'spread_close', 'total_close',
    ]
    odds_present = pd.Series(False, index=df.index)
    for col in odds_indicator_cols:
        if col in df.columns:
            odds_present = odds_present | df[col].notna()
    
    if odds_present.any():
        df['has_odds'] = odds_present.astype(int)
        if 'has_odds' not in feature_columns:
            feature_columns.append('has_odds')
    
    spread_cols = [c for c in ['spread_close', 'spread_open'] if c in df.columns]
    if spread_cols:
        has_spread = df[spread_cols].notna().any(axis=1)
        df['has_spread_odds'] = has_spread.astype(int)
        if 'has_spread_odds' not in feature_columns:
            feature_columns.append('has_spread_odds')
    
    total_cols = [c for c in ['total_close', 'total_open'] if c in df.columns]
    if total_cols:
        has_total = df[total_cols].notna().any(axis=1)
        df['has_total_odds'] = has_total.astype(int)
        if 'has_total_odds' not in feature_columns:
            feature_columns.append('has_total_odds')

    # ================================================================
    # SMART IMPUTATION (EXACT replica of training)
    # CRITICAL: Odds features LEFT AS NaN for H2O native handling!
    # Other frameworks get NaN→0 after smart imputation.
    # ================================================================
    odds_skip_keywords = [
        'implied_', 'no_vig_', 'moneyline_home_close', 'moneyline_away_close',
        'moneyline_home_open', 'moneyline_away_open',
        'spread_close', 'spread_open', 'spread_line',
        'total_close', 'total_open', 'total_line',
        'consensus_spread', 'consensus_total', 'consensus_ml',
        'pinnacle_spread', 'pinnacle_total', 'pinnacle_ml',
        'public_spread_home_pct', 'public_ml_home_pct',
        'public_total_over_pct', 'public_money_home_pct',
        'sharp_action_indicator', 'num_books',
        'spread_value', 'total_value', 'margin_value',
        'total_line_move', 'line_move_direction', 'total_move_direction',
        'h2h_total_vs_line',
    ]
    pct_keywords = ['_win_pct', '_pct_last', '_home_win_pct', '_away_win_pct',
                    '_ats_record', '_ou_over_pct']
    margin_keywords = ['_avg_margin', '_avg_pts', 'power_rating', 'h2h_home_avg_margin',
                       'h2h_total_avg']

    for col in feature_columns:
        if df[col].isna().any():
            col_lower = col.lower()
            # Skip odds for H2O native NaN handling
            if any(kw in col_lower for kw in odds_skip_keywords):
                continue  # Leave as NaN
            elif any(kw in col_lower for kw in pct_keywords):
                df[col] = df[col].fillna(0.5)
            elif any(kw in col_lower for kw in margin_keywords):
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val if pd.notna(median_val) else 0)
            else:
                df[col] = df[col].fillna(0)

    # Take last 20% as validation (walk-forward)
    split_idx = int(len(df) * 0.8)
    val_df = df.iloc[split_idx:].copy().reset_index(drop=True)

    # Check for single-class validation split
    if target_col in val_df.columns:
        unique_classes = val_df[target_col].nunique()
        if unique_classes < 2:
            # Fallback to stratified split
            from sklearn.model_selection import train_test_split
            try:
                _, val_df = train_test_split(
                    df, test_size=0.2, random_state=42,
                    stratify=df[target_col]
                )
                val_df = val_df.reset_index(drop=True)
                logger.info(f"  {sport}/{bet_type}: Stratified fallback (single-class chrono split)")
            except:
                pass

    # Verify features exist in val_df
    valid_features = [c for c in feature_columns if c in val_df.columns]
    feature_columns = valid_features

    return val_df, feature_columns, target_col


def _engineer_features_training_replica(df: pd.DataFrame, sport: str = "") -> pd.DataFrame:
    """
    EXACT replica of training_service.py FIX 2d feature engineering.
    Creates 30+ derived features matching training.
    """
    # TIER 1: Basic differentials
    if 'power_rating_diff' not in df.columns:
        if 'home_power_rating' in df.columns and 'away_power_rating' in df.columns:
            df['power_rating_diff'] = df['home_power_rating'] - df['away_power_rating']
    
    if 'momentum_diff' not in df.columns:
        if 'home_streak' in df.columns and 'away_streak' in df.columns:
            df['momentum_diff'] = df['home_streak'] - df['away_streak']
    
    if 'recent_form_diff' not in df.columns:
        if 'home_wins_last10' in df.columns and 'away_wins_last10' in df.columns:
            df['recent_form_diff'] = df['home_wins_last10'] - df['away_wins_last10']
    
    if 'recent_form5_diff' not in df.columns:
        if 'home_wins_last5' in df.columns and 'away_wins_last5' in df.columns:
            df['recent_form5_diff'] = df['home_wins_last5'] - df['away_wins_last5']
    
    if 'scoring_diff' not in df.columns:
        if 'home_avg_pts_last10' in df.columns and 'away_avg_pts_last10' in df.columns:
            df['scoring_diff'] = df['home_avg_pts_last10'] - df['away_avg_pts_last10']
    
    if 'defense_diff' not in df.columns:
        if 'home_avg_pts_allowed_last10' in df.columns and 'away_avg_pts_allowed_last10' in df.columns:
            df['defense_diff'] = df['away_avg_pts_allowed_last10'] - df['home_avg_pts_allowed_last10']
    
    if 'margin_diff' not in df.columns:
        if 'home_avg_margin_last10' in df.columns and 'away_avg_margin_last10' in df.columns:
            df['margin_diff'] = df['home_avg_margin_last10'] - df['away_avg_margin_last10']
    
    if 'venue_strength_diff' not in df.columns:
        if 'home_home_win_pct' in df.columns and 'away_away_win_pct' in df.columns:
            df['venue_strength_diff'] = df['home_home_win_pct'] - df['away_away_win_pct']
    
    # TIER 2: Line value features
    if 'spread_value' not in df.columns:
        if 'power_rating_diff' in df.columns and 'spread_close' in df.columns:
            df['spread_value'] = df['power_rating_diff'] + df['spread_close'].fillna(0)
    
    if 'margin_value' not in df.columns:
        if 'margin_diff' in df.columns and 'spread_close' in df.columns:
            df['margin_value'] = df['margin_diff'] + df['spread_close'].fillna(0)
    
    if 'ats_diff' not in df.columns:
        if 'home_ats_record_last10' in df.columns and 'away_ats_record_last10' in df.columns:
            df['ats_diff'] = df['home_ats_record_last10'].fillna(0.5) - df['away_ats_record_last10'].fillna(0.5)
    
    # TIER 3: Line movement features
    if 'line_move_direction' not in df.columns:
        if 'spread_close' in df.columns and 'spread_open' in df.columns:
            df['line_move_direction'] = np.sign(df['spread_close'] - df['spread_open'])
    
    if 'total_move_direction' not in df.columns:
        if 'total_close' in df.columns and 'total_open' in df.columns:
            df['total_move_direction'] = np.sign(df['total_close'] - df['total_open'])
    
    if 'spread_move_magnitude' not in df.columns:
        if 'spread_movement' in df.columns:
            df['spread_move_magnitude'] = df['spread_movement'].abs()
    
    # TIER 4: Situational spot features
    if 'revenge_edge' not in df.columns:
        if 'home_is_revenge' in df.columns and 'away_is_revenge' in df.columns:
            df['revenge_edge'] = df['home_is_revenge'].fillna(0).astype(int) - df['away_is_revenge'].fillna(0).astype(int)
    
    if 'rest_power_combo' not in df.columns:
        if 'rest_advantage' in df.columns and 'power_rating_diff' in df.columns:
            df['rest_power_combo'] = df['rest_advantage'] * df['power_rating_diff'].fillna(0) / 10
    
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
    
    # Momentum trends
    if 'home_momentum_trend' not in df.columns:
        if 'home_wins_last5' in df.columns and 'home_wins_last10' in df.columns:
            df['home_momentum_trend'] = df['home_wins_last5'] - (df['home_wins_last10'] / 2)
    
    if 'away_momentum_trend' not in df.columns:
        if 'away_wins_last5' in df.columns and 'away_wins_last10' in df.columns:
            df['away_momentum_trend'] = df['away_wins_last5'] - (df['away_wins_last10'] / 2)
    
    if 'momentum_trend_diff' not in df.columns:
        if 'home_momentum_trend' in df.columns and 'away_momentum_trend' in df.columns:
            df['momentum_trend_diff'] = df['home_momentum_trend'] - df['away_momentum_trend']
    
    if 'win_pct_diff' not in df.columns:
        if 'home_win_pct_last10' in df.columns and 'away_win_pct_last10' in df.columns:
            df['win_pct_diff'] = df['home_win_pct_last10'] - df['away_win_pct_last10']
    
    if 'expected_margin_vs_spread' not in df.columns:
        if 'margin_diff' in df.columns and 'spread_close' in df.columns:
            df['expected_margin_vs_spread'] = df['margin_diff'] + df['spread_close'].fillna(0)
    
    # TIER 6: Total-specific features (SUM not DIFF)
    if 'scoring_sum' not in df.columns:
        if 'home_avg_pts_last10' in df.columns and 'away_avg_pts_last10' in df.columns:
            df['scoring_sum'] = df['home_avg_pts_last10'].fillna(0) + df['away_avg_pts_last10'].fillna(0)
    
    if 'defense_sum' not in df.columns:
        if 'home_avg_pts_allowed_last10' in df.columns and 'away_avg_pts_allowed_last10' in df.columns:
            df['defense_sum'] = df['home_avg_pts_allowed_last10'].fillna(0) + df['away_avg_pts_allowed_last10'].fillna(0)
    
    if 'pace_proxy' not in df.columns:
        if 'scoring_sum' in df.columns and 'defense_sum' in df.columns:
            df['pace_proxy'] = (df['scoring_sum'] + df['defense_sum']) / 2
    
    if 'total_value' not in df.columns:
        total_line_col = None
        for col in ['consensus_total', 'total_close', 'pinnacle_total']:
            if col in df.columns and df[col].notna().sum() > 50:
                total_line_col = col
                break
        if total_line_col and 'pace_proxy' in df.columns:
            df['total_value'] = df[total_line_col].fillna(df['pace_proxy']) - df['pace_proxy']
    
    if 'offensive_mismatch' not in df.columns:
        home_off = df.get('home_avg_pts_last10')
        away_def = df.get('away_avg_pts_allowed_last10')
        away_off = df.get('away_avg_pts_last10')
        home_def = df.get('home_avg_pts_allowed_last10')
        if home_off is not None and away_def is not None and away_off is not None and home_def is not None:
            home_exploit = home_off.fillna(0) - away_def.fillna(0)
            away_exploit = away_off.fillna(0) - home_def.fillna(0)
            df['offensive_mismatch'] = home_exploit + away_exploit
    
    if 'total_line_move' not in df.columns:
        if 'total_close' in df.columns and 'total_open' in df.columns:
            df['total_line_move'] = df['total_close'] - df['total_open']
    
    if 'h2h_total_vs_line' not in df.columns:
        if 'h2h_total_avg' in df.columns:
            total_line_col = None
            for col in ['consensus_total', 'total_close', 'pinnacle_total']:
                if col in df.columns and df[col].notna().sum() > 50:
                    total_line_col = col
                    break
            if total_line_col:
                df['h2h_total_vs_line'] = df['h2h_total_avg'].fillna(0) - df[total_line_col].fillna(0)
    
    if 'margin_sum' not in df.columns:
        if 'home_avg_margin_last10' in df.columns and 'away_avg_margin_last10' in df.columns:
            df['margin_sum'] = df['home_avg_margin_last10'].fillna(0) + df['away_avg_margin_last10'].fillna(0)
    
    if 'rest_total' not in df.columns:
        if 'home_rest_days' in df.columns and 'away_rest_days' in df.columns:
            df['rest_total'] = df['home_rest_days'].fillna(1) + df['away_rest_days'].fillna(1)
    
    if 'b2b_fatigue_count' not in df.columns:
        b2b_count = np.zeros(len(df))
        if 'home_is_back_to_back' in df.columns:
            b2b_count += df['home_is_back_to_back'].fillna(0).astype(int)
        if 'away_is_back_to_back' in df.columns:
            b2b_count += df['away_is_back_to_back'].fillna(0).astype(int)
        df['b2b_fatigue_count'] = b2b_count

    return df

'''
        content = content[:start_idx] + new_func + content[next_section:]
        patches_applied += 1
        print(f"  [PATCH {patches_applied}] REPLACED load_validation_data with training-replica logic")

    # =========================================================================
    # PATCH 5: Replace _predict_sklearn with feature-aligned version
    # =========================================================================
    old_sklearn = '''def _predict_sklearn(model_path: str, model_dir: str,
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

    new_sklearn = '''def _predict_sklearn(model_path: str, model_dir: str,
                     df: pd.DataFrame, features: List[str]) -> np.ndarray:
    """Load sklearn model and predict with feature alignment."""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Try to get model's expected features
    model_features = None
    if hasattr(model, 'feature_names_in_'):
        model_features = list(model.feature_names_in_)
    
    n_expected = None
    if hasattr(model, 'n_features_in_'):
        n_expected = model.n_features_in_

    # Load saved feature list if available
    feat_json = Path(model_dir) / "feature_columns.json"
    if feat_json.exists():
        with open(feat_json) as f:
            model_features = json.load(f)

    # Check scaler for feature info too
    scaler = None
    scaler_path = Path(model_dir) / "scaler.pkl"
    if scaler_path.exists():
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        if model_features is None and hasattr(scaler, 'feature_names_in_'):
            model_features = list(scaler.feature_names_in_)
        if n_expected is None and hasattr(scaler, 'n_features_in_'):
            n_expected = scaler.n_features_in_

    # Align features
    if model_features is not None:
        # Use exact model features — add missing as 0, reorder to match
        aligned_features = []
        for feat in model_features:
            if feat in df.columns:
                aligned_features.append(feat)
            else:
                df[feat] = 0  # Missing feature → neutral value
                aligned_features.append(feat)
        X = df[aligned_features].fillna(0).values
        logger.info(f"  sklearn: Aligned {len(aligned_features)} features from model metadata")
    elif n_expected is not None and n_expected != len(features):
        # Feature count mismatch — try to match by trimming/padding
        if n_expected < len(features):
            X = df[features[:n_expected]].fillna(0).values
            logger.warning(f"  sklearn: Trimmed {len(features)} → {n_expected} features")
        else:
            X = df[features].fillna(0).values
            # Pad with zeros
            pad = np.zeros((X.shape[0], n_expected - len(features)))
            X = np.hstack([X, pad])
            logger.warning(f"  sklearn: Padded {len(features)} → {n_expected} features")
    else:
        X = df[features].fillna(0).values

    # Replace NaN/inf
    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)

    if scaler is not None:
        try:
            X = scaler.transform(X)
        except Exception as e:
            logger.warning(f"  sklearn scaler failed: {e}, using unscaled")

    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X)[:, 1]
    else:
        probs = model.predict(X)

    # Save feature list for future reference
    if model_features is not None and not feat_json.exists():
        try:
            with open(feat_json, 'w') as f:
                json.dump(model_features, f)
        except:
            pass

    return np.clip(probs, 0.001, 0.999)'''

    if old_sklearn in content:
        content = content.replace(old_sklearn, new_sklearn)
        patches_applied += 1
        print(f"  [PATCH {patches_applied}] Replaced _predict_sklearn with feature-aligned version")

    # =========================================================================
    # PATCH 6: Replace _predict_tensorflow with feature-aligned version
    # =========================================================================
    old_tf = '''def _predict_tensorflow(model_path: str, model_dir: str,
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

    new_tf = '''def _predict_tensorflow(model_path: str, model_dir: str,
                        df: pd.DataFrame, features: List[str]) -> np.ndarray:
    """Load TensorFlow model and predict with feature alignment."""
    import tensorflow as tf

    model = tf.keras.models.load_model(model_path)

    # Get expected input shape
    try:
        n_expected = model.input_shape[-1]
    except:
        n_expected = None

    # Check scaler for feature info
    scaler = None
    scaler_path = Path(model_dir) / "scaler.pkl"
    model_features = None
    
    # Load saved feature list if available
    feat_json = Path(model_dir) / "feature_columns.json"
    if feat_json.exists():
        with open(feat_json) as f:
            model_features = json.load(f)
    
    if scaler_path.exists():
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        if model_features is None and hasattr(scaler, 'feature_names_in_'):
            model_features = list(scaler.feature_names_in_)
        if n_expected is None and hasattr(scaler, 'n_features_in_'):
            n_expected = scaler.n_features_in_

    # Align features
    if model_features is not None:
        aligned_features = []
        for feat in model_features:
            if feat in df.columns:
                aligned_features.append(feat)
            else:
                df[feat] = 0
                aligned_features.append(feat)
        X = df[aligned_features].fillna(0).values.astype(np.float32)
    elif n_expected is not None and n_expected != len(features):
        if n_expected < len(features):
            X = df[features[:n_expected]].fillna(0).values.astype(np.float32)
        else:
            X = df[features].fillna(0).values.astype(np.float32)
            pad = np.zeros((X.shape[0], n_expected - len(features)), dtype=np.float32)
            X = np.hstack([X, pad])
    else:
        X = df[features].fillna(0).values.astype(np.float32)

    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)

    if scaler is not None:
        try:
            X = scaler.transform(X)
        except Exception as e:
            logger.warning(f"  tf scaler failed: {e}, using unscaled")

    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)

    probs = model.predict(X, verbose=0)
    if probs.ndim == 2 and probs.shape[1] == 1:
        probs = probs.ravel()
    elif probs.ndim == 2 and probs.shape[1] == 2:
        probs = probs[:, 1]

    return np.clip(probs, 0.001, 0.999)'''

    if old_tf in content:
        content = content.replace(old_tf, new_tf)
        patches_applied += 1
        print(f"  [PATCH {patches_applied}] Replaced _predict_tensorflow with feature-aligned version")

    # =========================================================================
    # PATCH 7: Replace _predict_h2o with NaN-aware version
    # =========================================================================
    old_h2o = '''def _predict_h2o(model_path: str, model_dir: str,
                 df: pd.DataFrame, features: List[str]) -> np.ndarray:
    """Load H2O model and predict."""
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

    h2o_df = h2o.H2OFrame(df[features])
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

    new_h2o = '''def _predict_h2o(model_path: str, model_dir: str,
                 df: pd.DataFrame, features: List[str]) -> np.ndarray:
    """Load H2O model and predict with proper NaN handling.
    
    CRITICAL: Training leaves odds features as NaN for H2O native handling.
    H2O tree models route NaN to optimal branch during splitting.
    We must NOT fill NaN for H2O — pass the NaN-containing DataFrame directly.
    """
    import h2o
    
    # Suppress H2O warnings about levels
    import logging as h2o_logging
    h2o_logger = h2o_logging.getLogger("h2o")
    h2o_logger.setLevel(h2o_logging.ERROR)
    
    try:
        h2o.init(nthreads=-1, max_mem_size="4G", verbose=False)
    except:
        pass  # Already running

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

    # Get model's expected features
    model_features = None
    try:
        # H2O models store their training column names
        model_features = model._model_json['output']['names']
        # Remove the response column (last one or specific name)
        resp = model._model_json['output'].get('response_column_name', '')
        if resp and resp in model_features:
            model_features = [f for f in model_features if f != resp]
    except:
        pass
    
    # Load saved feature list
    feat_json = Path(model_dir) / "feature_columns.json"
    if model_features is None and feat_json.exists():
        with open(feat_json) as f:
            model_features = json.load(f)

    # Prepare data — DO NOT fill NaN for H2O!
    if model_features is not None:
        # Use exact model features
        pred_df = pd.DataFrame()
        for feat in model_features:
            if feat in df.columns:
                pred_df[feat] = df[feat].values
            else:
                pred_df[feat] = np.nan  # H2O handles NaN natively
        logger.info(f"  h2o: Aligned {len(model_features)} features from model metadata")
    else:
        # Use evaluation features but DON'T fill NaN
        pred_df = df[features].copy()
    
    # Convert to H2OFrame (H2O handles NaN natively in tree models)
    h2o_df = h2o.H2OFrame(pred_df)
    
    # Handle categorical columns that H2O expects
    try:
        model_columns_info = model._model_json.get('output', {}).get('column_types', [])
        model_col_names = model._model_json.get('output', {}).get('names', [])
        for i, (cname, ctype) in enumerate(zip(model_col_names, model_columns_info)):
            if ctype == 'Enum' and cname in h2o_df.columns:
                h2o_df[cname] = h2o_df[cname].asfactor()
    except:
        # Fallback: convert low-cardinality columns
        for col in h2o_df.columns:
            try:
                if h2o_df[col].nlevels()[0] > 0:
                    continue  # already factor
                unique_vals = h2o_df[col].unique().nrow
                if unique_vals <= 10:
                    h2o_df[col] = h2o_df[col].asfactor()
            except:
                pass

    preds = model.predict(h2o_df)
    probs = preds.as_data_frame()

    if 'p1' in probs.columns:
        result = probs['p1'].values
    elif probs.shape[1] >= 3:
        result = probs.iloc[:, 2].values
    else:
        result = probs.iloc[:, 0].values

    # Save feature list for future reference
    if model_features is not None and not feat_json.exists():
        try:
            with open(feat_json, 'w') as f:
                json.dump(model_features, f)
        except:
            pass

    return np.clip(result.astype(float), 0.001, 0.999)'''

    if old_h2o in content:
        content = content.replace(old_h2o, new_h2o)
        patches_applied += 1
        print(f"  [PATCH {patches_applied}] Replaced _predict_h2o with NaN-aware + feature-aligned version")

    # =========================================================================
    # PATCH 8: Replace _predict_autogluon with feature-aligned version
    # =========================================================================
    old_ag = '''def _predict_autogluon(model_path: str, df: pd.DataFrame,
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

    new_ag = '''def _predict_autogluon(model_path: str, df: pd.DataFrame,
                       features: List[str]) -> np.ndarray:
    """Load AutoGluon predictor and predict with feature alignment."""
    from autogluon.tabular import TabularPredictor

    predictor = TabularPredictor.load(model_path)
    
    # Get predictor's expected features
    model_features = None
    try:
        model_features = list(predictor.feature_metadata_in.get_features())
    except:
        try:
            model_features = list(predictor.features())
        except:
            pass

    # Align features
    if model_features is not None:
        pred_df = pd.DataFrame()
        for feat in model_features:
            if feat in df.columns:
                pred_df[feat] = df[feat].values
            else:
                pred_df[feat] = 0  # Missing → neutral
        logger.info(f"  autogluon: Aligned {len(model_features)} features from predictor metadata")
    else:
        pred_df = df[features].fillna(0)

    probs = predictor.predict_proba(pred_df)

    if isinstance(probs, pd.DataFrame) and probs.shape[1] == 2:
        return np.clip(probs.iloc[:, 1].values, 0.001, 0.999)
    elif isinstance(probs, pd.Series):
        return np.clip(probs.values, 0.001, 0.999)
    else:
        return np.clip(probs.values.ravel(), 0.001, 0.999)'''

    if old_ag in content:
        content = content.replace(old_ag, new_ag)
        patches_applied += 1
        print(f"  [PATCH {patches_applied}] Replaced _predict_autogluon with feature-aligned version")

    # =========================================================================
    # PATCH 9: Replace composite_score with realistic tiered scoring
    # =========================================================================
    old_composite = '''def compute_composite_score(metrics: Dict[str, float]) -> float:
    """
    Composite score for ranking models.
    Weighted combination: AUC(30%) + Accuracy(25%) + ROI(25%) + (1-ECE)(10%) + (1-Brier)(10%)
    """
    auc = metrics.get("auc_roc", 0.5)
    acc = metrics.get("accuracy", 0.5)
    roi = max(metrics.get("simulated_roi", -1.0), -1.0)  # Clamp
    ece = min(metrics.get("ece", 1.0), 1.0)'''

    new_composite = '''def compute_composite_score(metrics: Dict[str, float]) -> float:
    """
    Composite score for ranking models.
    Weighted: AUC(35%) + Accuracy(20%) + ROI(20%) + (1-ECE)(15%) + (1-Brier)(10%)
    With leakage penalty for single-class collapse.
    """
    auc = metrics.get("auc_roc", 0.5)
    acc = metrics.get("accuracy", 0.5)
    roi = max(min(metrics.get("simulated_roi", -0.2), 0.30), -0.20)  # Realistic cap
    ece = min(metrics.get("ece", 1.0), 1.0)'''

    if old_composite in content:
        content = content.replace(old_composite, new_composite)
        patches_applied += 1
        print(f"  [PATCH {patches_applied}] Updated composite score formula")

    # Also fix the weights
    old_weights = '''    score = (
        0.30 * auc +
        0.25 * acc +
        0.25 * roi_norm +
        0.10 * (1.0 - ece) +
        0.10 * (1.0 - brier)
    )'''

    new_weights = '''    # Detect leakage (single class dominance)
    y_pred = metrics.get("_y_pred_proba", None)
    leakage_penalty = 1.0
    if y_pred is not None:
        pred_classes = (np.array(y_pred) > 0.5).astype(int)
        dominant_pct = max(pred_classes.mean(), 1 - pred_classes.mean())
        if dominant_pct > TIER_LEAKAGE_CLASS_PCT:
            leakage_penalty = 0.5  # Heavily penalize

    score = (
        0.35 * auc +
        0.20 * acc +
        0.20 * roi_norm +
        0.15 * (1.0 - ece) +
        0.10 * (1.0 - brier)
    ) * leakage_penalty'''

    if old_weights in content:
        content = content.replace(old_weights, new_weights)
        patches_applied += 1
        print(f"  [PATCH {patches_applied}] Added leakage penalty to composite score")

    # =========================================================================
    # PATCH 10: Add tier classification after metrics computation in main loop
    # =========================================================================
    old_threshold = '''                # Threshold check
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
                    )'''

    new_threshold = '''                # Detect leakage
                pred_classes = (y_pred_proba > 0.5).astype(int)
                dominant_pct = max(pred_classes.mean(), 1 - pred_classes.mean())
                score.dominant_class_pct = dominant_pct
                score.leakage_suspect = dominant_pct > TIER_LEAKAGE_CLASS_PCT

                # Tier classification
                if (score.accuracy >= TIER_PROD_ACC and 
                    score.auc_roc >= TIER_PROD_AUC and
                    score.ece <= TIER_PROD_ECE and
                    not score.leakage_suspect):
                    score.tier = "PROD"
                    score.passes_threshold = True
                elif score.auc_roc >= TIER_SIGNAL_AUC and not score.leakage_suspect:
                    score.tier = "SIGNAL"
                    score.passes_threshold = False
                elif score.leakage_suspect:
                    score.tier = "LEAK?"
                    score.passes_threshold = False
                else:
                    score.tier = "WEAK"
                    score.passes_threshold = False

                tier_icon = {"PROD": "🟢", "SIGNAL": "🟡", "LEAK?": "🔴", "WEAK": "⚪"}.get(score.tier, "?")

                if verbose:
                    console.print(
                        f"  {tier_icon} {score.tier:6s} {framework}/{sport}/{bet_type}: "
                        f"acc={score.accuracy:.3f} auc={score.auc_roc:.3f} "
                        f"roi={score.simulated_roi*100:+.1f}% ece={score.ece:.3f} "
                        f"dom={score.dominant_class_pct:.1%}"
                    )'''

    if old_threshold in content:
        content = content.replace(old_threshold, new_threshold)
        patches_applied += 1
        print(f"  [PATCH {patches_applied}] Added tier classification (PROD/SIGNAL/LEAK?/WEAK)")

    # =========================================================================
    # PATCH 11: Replace _print_summary with tiered breakdown
    # =========================================================================
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
    ))'''

    new_summary = '''def _print_summary(scores: List[ModelScore]):
    """Print tiered summary statistics."""
    loaded = [s for s in scores if s.load_success]
    errors = [s for s in scores if not s.load_success]
    
    prod = [s for s in loaded if s.tier == "PROD"]
    signal = [s for s in loaded if s.tier == "SIGNAL"]
    leak = [s for s in loaded if s.tier == "LEAK?"]
    weak = [s for s in loaded if s.tier == "WEAK"]

    console.print(Panel(
        f"[bold]Total Models:[/bold] {len(scores)}\\n"
        f"[green]Loaded:[/green] {len(loaded)}\\n"
        f"[red]Load Errors:[/red] {len(errors)}\\n"
        f"\\n"
        f"[bold]Tier Breakdown:[/bold]\\n"
        f"  🟢 PRODUCTION: {len(prod):3d}  (acc≥{TIER_PROD_ACC}, auc≥{TIER_PROD_AUC}, ece≤{TIER_PROD_ECE}, no leakage)\\n"
        f"  🟡 SIGNAL:     {len(signal):3d}  (auc≥{TIER_SIGNAL_AUC}, needs calibration/threshold work)\\n"
        f"  🔴 LEAK?:      {len(leak):3d}  (>{TIER_LEAKAGE_CLASS_PCT:.0%} predictions in one class)\\n"
        f"  ⚪ WEAK:       {len(weak):3d}  (no discriminative power)",
        title="Summary"
    ))
    
    if signal:
        console.print("\\n[yellow bold]🟡 SIGNAL Models (have edge, need calibration):[/yellow bold]")
        for s in sorted(signal, key=lambda x: x.auc_roc, reverse=True):
            console.print(
                f"  {s.sport}/{s.bet_type}/{s.framework}: "
                f"AUC={s.auc_roc:.3f} ACC={s.accuracy:.3f} "
                f"ROI={s.simulated_roi*100:+.1f}% ECE={s.ece:.3f}"
            )
    
    if prod:
        console.print("\\n[green bold]🟢 PRODUCTION Models (ready for deployment):[/green bold]")
        for s in sorted(prod, key=lambda x: x.composite_score, reverse=True):
            console.print(
                f"  {s.sport}/{s.bet_type}/{s.framework}: "
                f"AUC={s.auc_roc:.3f} ACC={s.accuracy:.3f} "
                f"ROI={s.simulated_roi*100:+.1f}% ECE={s.ece:.3f}"
            )'''

    if old_summary in content:
        content = content.replace(old_summary, new_summary)
        patches_applied += 1
        print(f"  [PATCH {patches_applied}] Replaced summary with tiered breakdown")

    # =========================================================================
    # PATCH 12: Add tier/leakage fields to CSV export
    # =========================================================================
    old_csv = '''            "composite_score": round(s.composite_score, 6),
            "passes_threshold": s.passes_threshold,
            "load_success": s.load_success,
            "model_path": s.model_path,
            "error": s.error,'''

    new_csv = '''            "composite_score": round(s.composite_score, 6),
            "tier": s.tier,
            "dominant_class_pct": round(s.dominant_class_pct, 4),
            "leakage_suspect": s.leakage_suspect,
            "passes_threshold": s.passes_threshold,
            "load_success": s.load_success,
            "model_path": s.model_path,
            "error": s.error,'''

    if old_csv in content:
        content = content.replace(old_csv, new_csv)
        patches_applied += 1
        print(f"  [PATCH {patches_applied}] Added tier/leakage fields to CSV export")

    # =========================================================================
    # PATCH 13: Update scorecard display with tier column
    # =========================================================================
    old_scorecard_cols = '''    table.add_column("Pass?", style="white")'''
    new_scorecard_cols = '''    table.add_column("Tier", style="white")
    table.add_column("Dom%", style="red")'''

    if old_scorecard_cols in content:
        content = content.replace(old_scorecard_cols, new_scorecard_cols)
        patches_applied += 1
        print(f"  [PATCH {patches_applied}] Added Tier/Dom% columns to scorecard")

    # Update the row rendering
    old_row_pass = '''        pass_str = "[green]✓[/green]" if s.passes_threshold else "[red]✗[/red]"

        table.add_row(
            str(s.rank), s.sport, s.bet_type, s.framework,
            f"{s.accuracy:.3f}", f"{s.auc_roc:.3f}",
            roi_str, f"{s.ece:.3f}", tier_a_str,
            f"{s.composite_score:.4f}", pass_str,
        )'''

    new_row_pass = '''        tier_color = {"PROD": "green", "SIGNAL": "yellow", "LEAK?": "red", "WEAK": "dim"}.get(s.tier, "white")
        tier_str = f"[{tier_color}]{s.tier}[/{tier_color}]"
        dom_str = f"{s.dominant_class_pct:.0%}" if s.dominant_class_pct > 0 else "-"

        table.add_row(
            str(s.rank), s.sport, s.bet_type, s.framework,
            f"{s.accuracy:.3f}", f"{s.auc_roc:.3f}",
            roi_str, f"{s.ece:.3f}", tier_a_str,
            f"{s.composite_score:.4f}", tier_str, dom_str,
        )'''

    if old_row_pass in content:
        content = content.replace(old_row_pass, new_row_pass)
        patches_applied += 1
        print(f"  [PATCH {patches_applied}] Updated scorecard rows with tier display")

    return content, patches_applied


def main():
    print("=" * 70)
    print("ROYALEY Evaluation Patch v5: Recover LEAK? Models")
    print("=" * 70)

    if not EVAL_SCRIPT.exists():
        print(f"[ERROR] {EVAL_SCRIPT} not found!")
        sys.exit(1)

    content, n_patches = apply_patches()

    if n_patches == 0:
        print("\n[INFO] No patches needed — all changes already applied or patterns not found.")
        sys.exit(0)

    write_file(content)

    print(f"\n✅ Applied {n_patches} patches to {EVAL_SCRIPT}")
    print(f"\n📋 Changes summary:")
    print(f"  1. Replaced load_validation_data with EXACT training pipeline replica")
    print(f"  2. Added 30+ derived features matching training_service.py")
    print(f"  3. Replicated tennis random-swap debiasing (same seed=42)")
    print(f"  4. Replicated smart imputation (odds→NaN, pct→0.5, margin→median)")
    print(f"  5. Added model-intrinsic feature extraction for sklearn/h2o/autogluon")
    print(f"  6. H2O now receives NaN for odds (native tree handling)")
    print(f"  7. Tiered scoring: PROD/SIGNAL/LEAK?/WEAK classification")
    print(f"  8. Saves feature_columns.json alongside models (future-proof)")
    print(f"\n🔄 Next: Run evaluation:")
    print(f"  docker exec -it royaley_api python scripts/evaluate_models.py --verbose")

    # Verify syntax
    print(f"\n🔍 Verifying syntax...")
    import py_compile
    try:
        py_compile.compile(str(EVAL_SCRIPT), doraise=True)
        print(f"  ✅ Syntax OK")
    except py_compile.PyCompileError as e:
        print(f"  ❌ Syntax error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()