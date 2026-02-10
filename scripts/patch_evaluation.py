#!/usr/bin/env python3
"""
ROYALEY Evaluation Patch v5.1 HOTFIX
Fixes 3 critical bugs in v5 patch:

BUG 1 (ROOT CAUSE of ALL dom=100%):
  load_validation_data uses pd.concat (vertical stack) for multiple CSVs,
  but training uses _load_from_csv which MERGES horizontally on game_id.
  Result: 7x rows, each row has only one file-type's columns, rest NaN.
  
BUG 2 (H2O 26 model load failures):
  Boolean columns (away_is_revenge, away_3_in_4_nights, etc.) are enum in 
  trained H2O models but numeric in evaluation data. Need to convert to 
  string BEFORE creating H2OFrame.

BUG 3 (crash at end):
  _print_summary replacement only covered first half of function, leaving 
  orphaned `if passed:` code.

Apply: docker cp patch_v5_1_hotfix.py royaley_api:/app/scripts/
       docker exec royaley_api python /app/scripts/patch_v5_1_hotfix.py
       docker exec -it royaley_api python scripts/evaluate_models.py --verbose
"""

import re

EVAL_PATH = "/app/scripts/evaluate_models.py"

print("=" * 70)
print("ROYALEY Evaluation Patch v5.1 HOTFIX")
print("=" * 70)

with open(EVAL_PATH, "r") as f:
    content = f.read()

patches_applied = 0

# ==========================================================================
# FIX 1: Replace CSV loading with MERGE logic (replicates _load_from_csv)
# ==========================================================================
# The current code does: pd.concat(dfs, ignore_index=True)
# This STACKS rows vertically. But training MERGES columns horizontally.
# We need to replicate training_service._load_from_csv() merge logic.

# Find the CSV loading section inside load_validation_data
old_csv_loading = '''    for csv_path in csv_paths:
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
            break'''

new_csv_loading = '''    for csv_path in csv_paths:
        if csv_path is None or not csv_path.exists():
            continue

        sport_dir = csv_path / sport
        if not sport_dir.exists():
            # Try parent dir with pattern
            sport_dir = csv_path
        
        if not sport_dir.exists():
            continue
            
        # Find CSV files for this sport (exact match to avoid NBA/WNBA collision)
        import re as _re
        sport_pattern = _re.compile(
            rf'(^|[_\\-\\s]){_re.escape(sport)}([_\\-\\s\\.]|$)', _re.IGNORECASE
        )
        
        csv_files = sorted([f for f in sport_dir.glob(f"ml_features_{sport}*.csv")
                           if sport_pattern.search(f.stem)])
        if not csv_files:
            csv_files = sorted([f for f in sport_dir.glob("*.csv")
                               if sport_pattern.search(f.stem)])
        
        if not csv_files:
            continue
        
        # ============================================================
        # MERGE logic (replica of training_service._load_from_csv)
        # Training merges: game, odds, player, situation, target, team, weather
        # ============================================================
        file_types = {}
        for csv_file in csv_files:
            name = csv_file.name.lower()
            if '_target_' in name or name.endswith('_target.csv'):
                file_types['target'] = csv_file
            elif '_game_' in name:
                file_types['game'] = csv_file
            elif '_odds_' in name:
                file_types['odds'] = csv_file
            elif '_player_' in name:
                file_types['player'] = csv_file
            elif '_situation_' in name:
                file_types['situation'] = csv_file
            elif '_team_' in name:
                file_types['team'] = csv_file
            elif '_weather_' in name:
                file_types['weather'] = csv_file
            else:
                if 'main' not in file_types:
                    file_types['main'] = csv_file
        
        logger.info(f"  CSV files found: {[(k, v.name) for k, v in file_types.items()]}")
        
        # Load all DataFrames
        loaded_dfs = {}
        for ftype, fpath in file_types.items():
            try:
                loaded_dfs[ftype] = pd.read_csv(fpath)
                logger.info(f"  Loaded {ftype}: {len(loaded_dfs[ftype])} rows x {len(loaded_dfs[ftype].columns)} cols")
            except Exception as e:
                logger.warning(f"  Failed to load {ftype}: {e}")
        
        if not loaded_dfs:
            continue
        
        # If only ONE CSV file, use it directly (no merge needed)
        if len(loaded_dfs) == 1:
            df = list(loaded_dfs.values())[0]
            logger.info(f"  Single CSV: {len(df)} rows x {len(df.columns)} cols")
            break
        
        # Start with main or largest dataframe as base
        if 'main' in loaded_dfs:
            df = loaded_dfs['main'].copy()
            base_type = 'main'
        else:
            base_type = max(loaded_dfs.keys(), key=lambda k: len(loaded_dfs[k]))
            df = loaded_dfs[base_type].copy()
        
        logger.info(f"  Base: {base_type} ({len(df)} rows x {len(df.columns)} cols)")
        
        # Find merge key
        merge_key = None
        for key in ['game_id', 'match_id', 'id', 'index', 'game_date', 'date']:
            if key in df.columns:
                merge_key = key
                break
        
        # HORIZONTAL MERGE other files (same as training!)
        for ftype, other_df in loaded_dfs.items():
            if ftype == base_type or other_df is None:
                continue
            
            if merge_key and merge_key in other_df.columns:
                new_cols = [c for c in other_df.columns if c not in df.columns or c == merge_key]
                if len(new_cols) > 1:
                    try:
                        df = df.merge(other_df[new_cols], on=merge_key, how='left')
                        logger.info(f"  Merged {ftype}: now {len(df.columns)} cols")
                    except Exception as e:
                        logger.warning(f"  Merge {ftype} failed: {e}")
            elif len(df) == len(other_df):
                new_cols = [c for c in other_df.columns if c not in df.columns]
                if new_cols:
                    df = pd.concat([df, other_df[new_cols]], axis=1)
                    logger.info(f"  Concat {ftype}: now {len(df.columns)} cols")
        
        logger.info(f"  Final: {len(df)} rows x {len(df.columns)} cols")
        break'''

if old_csv_loading in content:
    content = content.replace(old_csv_loading, new_csv_loading)
    patches_applied += 1
    print(f"  [FIX {patches_applied}] Replaced CSV concat with horizontal MERGE logic")
else:
    # Try to find the concat pattern more flexibly
    # Maybe whitespace differs
    concat_count = content.count("pd.concat(dfs, ignore_index=True)")
    if concat_count > 0:
        print(f"  WARNING: Found {concat_count} pd.concat calls but couldn't match exact block")
        print(f"  Attempting line-level replacement...")
        
        # Find load_validation_data function and replace csv loading section
        lines = content.split('\n')
        in_func = False
        func_start = -1
        csv_section_start = -1
        csv_section_end = -1
        brace_depth = 0
        
        for i, line in enumerate(lines):
            if 'def load_validation_data(' in line:
                in_func = True
                func_start = i
            elif in_func:
                if 'for csv_path in csv_paths:' in line:
                    csv_section_start = i
                elif csv_section_start > 0 and 'if df is None' in line and csv_section_end < 0:
                    csv_section_end = i
                    break
        
        if csv_section_start > 0 and csv_section_end > 0:
            # Get indentation
            indent = '    '  # inside function
            replacement_lines = new_csv_loading.split('\n')
            
            new_lines = lines[:csv_section_start] + replacement_lines + [''] + lines[csv_section_end:]
            content = '\n'.join(new_lines)
            patches_applied += 1
            print(f"  [FIX {patches_applied}] Replaced CSV section by line range ({csv_section_start}-{csv_section_end})")
        else:
            print(f"  FAILED: Could not locate CSV loading section (start={csv_section_start}, end={csv_section_end})")
    else:
        print(f"  SKIPPED: CSV loading pattern not found (already patched?)")

# ==========================================================================
# FIX 2: H2O categorical type handling - convert bool columns BEFORE H2OFrame
# ==========================================================================
# The current patch tries asfactor() AFTER H2OFrame creation, but H2O errors
# occur during predict() because the frame types don't match model expectations.
# Fix: convert columns to string in pandas BEFORE H2OFrame creation.

old_h2o_convert = '''    # Convert to H2OFrame (H2O handles NaN natively in tree models)
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
                pass'''

new_h2o_convert = '''    # ============================================================
    # CRITICAL: Match column types to what model expects
    # Training data had boolean cols as True/False → H2O auto-detected as Enum
    # We must convert those columns to string BEFORE H2OFrame creation
    # so H2O auto-detects them as Enum (categorical) again
    # ============================================================
    enum_cols = set()
    try:
        model_col_types = model._model_json.get('output', {}).get('column_types', [])
        model_col_names_full = model._model_json.get('output', {}).get('names', [])
        resp_col = model._model_json.get('output', {}).get('response_column_name', '')
        for cname, ctype in zip(model_col_names_full, model_col_types):
            if ctype == 'Enum' and cname != resp_col and cname in pred_df.columns:
                enum_cols.add(cname)
    except Exception as e:
        logger.warning(f"  h2o: Could not read model column types: {e}")
    
    # Also detect likely boolean columns from the data itself
    for col in pred_df.columns:
        if col in enum_cols:
            continue
        col_lower = col.lower()
        # Boolean indicator columns that H2O likely saw as Enum during training
        bool_indicators = [
            'is_revenge', 'is_back_to_back', 'is_b2b', 'letdown_spot', 
            'lookahead_spot', '3_in_4_nights', '4_in_6_nights', 
            'is_playoff', 'is_neutral_site', 'is_home', 'is_away',
            'is_division', 'is_conference', 'is_rivalry',
        ]
        if any(ind in col_lower for ind in bool_indicators):
            unique_vals = pred_df[col].dropna().unique()
            if len(unique_vals) <= 3:  # Binary or ternary
                enum_cols.add(col)
    
    if enum_cols:
        logger.info(f"  h2o: Converting {len(enum_cols)} columns to categorical: {sorted(list(enum_cols))[:5]}...")
        for col in enum_cols:
            # Convert to string so H2O auto-detects as Enum
            # NaN stays as NaN (H2O handles natively)
            pred_df[col] = pred_df[col].apply(
                lambda x: str(int(x)) if pd.notna(x) else None
            )
    
    # Force numeric columns to float64 to avoid H2O type confusion
    for col in pred_df.columns:
        if col not in enum_cols:
            if pred_df[col].dtype == 'bool':
                pred_df[col] = pred_df[col].astype(float)
            elif pred_df[col].dtype == 'object':
                try:
                    pred_df[col] = pd.to_numeric(pred_df[col], errors='coerce')
                except:
                    pass
    
    # Build H2OFrame column_types hint
    h2o_col_types = {}
    for col in pred_df.columns:
        if col in enum_cols:
            h2o_col_types[col] = 'enum'
        else:
            h2o_col_types[col] = 'numeric'
    
    # Convert to H2OFrame with explicit column types
    h2o_df = h2o.H2OFrame(pred_df, column_types=h2o_col_types)'''

if old_h2o_convert in content:
    content = content.replace(old_h2o_convert, new_h2o_convert)
    patches_applied += 1
    print(f"  [FIX {patches_applied}] Fixed H2O categorical handling (convert BEFORE H2OFrame)")
else:
    # Try without exact whitespace
    simplified_old = "h2o_df = h2o.H2OFrame(pred_df)"
    if simplified_old in content:
        # More targeted: just replace the H2OFrame creation and the try/except block after it
        # Find the block
        idx = content.index(simplified_old)
        # Find the end of the try/except block (look for "preds = model.predict")
        predict_idx = content.index("preds = model.predict(h2o_df)", idx)
        old_block = content[idx:predict_idx]
        
        new_block = new_h2o_convert.lstrip().split("h2o_df = h2o.H2OFrame(pred_df)")
        # Keep everything before h2o_df creation, insert new logic, then continue
        content = content[:idx] + new_h2o_convert.strip() + "\n\n    " + content[predict_idx:]
        patches_applied += 1
        print(f"  [FIX {patches_applied}] Fixed H2O categorical handling (targeted replacement)")
    else:
        print(f"  SKIPPED: H2O convert pattern not found")

# ==========================================================================
# FIX 3: Fix _print_summary - replace entire function including "if passed:" tail
# ==========================================================================
# The v5 patch only replaced the first half of _print_summary, leaving 
# orphaned code that references undefined `passed` variable.

# Strategy: Find the function from "def _print_summary" to the next "def " 
# and replace the whole thing.

# First try exact match of the orphaned code
old_passed_block = '''    if passed:
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
            )

        console.print(best_table)'''

if old_passed_block in content:
    content = content.replace(old_passed_block, "")
    patches_applied += 1
    print(f"  [FIX {patches_applied}] Removed orphaned 'if passed:' block from _print_summary")
else:
    # Try to find and remove it with flexible whitespace
    if "if passed:" in content:
        # Find it and remove everything from "if passed:" to "console.print(best_table)"
        lines = content.split('\n')
        new_lines = []
        skip = False
        skip_depth = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped == "if passed:":
                skip = True
                skip_depth = len(line) - len(line.lstrip())
                continue
            if skip:
                if stripped == "" or (len(line) - len(line.lstrip()) > skip_depth):
                    continue  # Still inside the if block
                elif stripped.startswith("console.print(best_table)"):
                    skip = False
                    continue
                elif len(line) - len(line.lstrip()) <= skip_depth and stripped != "":
                    skip = False
                    new_lines.append(line)
            else:
                new_lines.append(line)
        
        content = '\n'.join(new_lines)
        patches_applied += 1
        print(f"  [FIX {patches_applied}] Removed orphaned 'if passed:' block (flexible match)")
    else:
        print(f"  SKIPPED: 'if passed:' not found (already fixed?)")

# ==========================================================================
# FIX 4: Add diagnostic logging for feature alignment quality
# ==========================================================================
# Add counters to sklearn/tf/autogluon to show how many features are 
# real vs padded with 0 (helps debug if models still show dom=100%)

# For sklearn: add a diagnostic log line after alignment
old_sklearn_log = '''        logger.info(f"  sklearn: Aligned {len(aligned_features)} features from model metadata")'''
new_sklearn_log = '''        found_in_data = sum(1 for f in model_features if f in df.columns)
        padded_zero = len(model_features) - found_in_data
        logger.info(f"  sklearn: Aligned {len(aligned_features)} features from model metadata "
                    f"({found_in_data} found, {padded_zero} padded with 0)")
        if padded_zero > found_in_data:
            logger.warning(f"  sklearn: ⚠️ MORE features padded than found! Model may predict poorly.")'''

if old_sklearn_log in content:
    content = content.replace(old_sklearn_log, new_sklearn_log)
    patches_applied += 1
    print(f"  [FIX {patches_applied}] Added sklearn feature alignment diagnostics")

# Same for H2O
old_h2o_log = '''        logger.info(f"  h2o: Aligned {len(model_features)} features from model metadata")'''
new_h2o_log = '''        found_in_data = sum(1 for f in model_features if f in df.columns)
        padded_nan = len(model_features) - found_in_data
        logger.info(f"  h2o: Aligned {len(model_features)} features from model metadata "
                    f"({found_in_data} found, {padded_nan} padded with NaN)")'''

if old_h2o_log in content:
    content = content.replace(old_h2o_log, new_h2o_log)
    patches_applied += 1
    print(f"  [FIX {patches_applied}] Added H2O feature alignment diagnostics")

# Same for autogluon - need to find exact indentation
ag_pattern = 'logger.info(f"  autogluon: Aligned {len(model_features)} features from predictor metadata")'
if ag_pattern in content:
    # Find it and determine indentation
    idx = content.index(ag_pattern)
    # Find start of this line
    line_start = content.rfind('\n', 0, idx) + 1
    indent = content[line_start:idx]  # whitespace before logger
    
    old_line = indent + ag_pattern
    new_lines = (
        indent + 'found_in_data = sum(1 for f in model_features if f in df.columns)\n' +
        indent + 'padded = len(model_features) - found_in_data\n' +
        indent + 'logger.info(f"  autogluon: Aligned {len(model_features)} features from predictor metadata "\n' +
        indent + '            f"({found_in_data} found, {padded} padded)")'
    )
    content = content.replace(old_line, new_lines)
    patches_applied += 1
    print(f"  [FIX {patches_applied}] Added autogluon feature alignment diagnostics")

# ==========================================================================
# FIX 5: Add data shape diagnostic after load_validation_data
# ==========================================================================
# Log the shape and non-null counts right after data loading
old_eval_load = '''                val_df, feature_columns, target_col = val_result
                score.n_validation_samples = len(val_df)
                score.n_features = len(feature_columns)'''

new_eval_load = '''                val_df, feature_columns, target_col = val_result
                score.n_validation_samples = len(val_df)
                score.n_features = len(feature_columns)
                
                # Diagnostic: check data quality
                if len(feature_columns) > 0:
                    non_null_pcts = val_df[feature_columns].notna().mean()
                    avg_non_null = non_null_pcts.mean() * 100
                    zero_pcts = (val_df[feature_columns] == 0).mean()
                    avg_zero = zero_pcts.mean() * 100
                    logger.info(f"  Data: {len(val_df)} rows x {len(feature_columns)} features, "
                               f"{avg_non_null:.0f}% non-null, {avg_zero:.0f}% zeros")'''

if old_eval_load in content:
    content = content.replace(old_eval_load, new_eval_load)
    patches_applied += 1
    print(f"  [FIX {patches_applied}] Added data quality diagnostics after loading")
else:
    print(f"  SKIPPED: Could not find eval load pattern for diagnostics")

# ==========================================================================
# Write patched file
# ==========================================================================
with open(EVAL_PATH, "w") as f:
    f.write(content)

print(f"\n✅ Applied {patches_applied} fixes to {EVAL_PATH}")

# Verify syntax
print("\n🔍 Verifying syntax...")
import py_compile
try:
    py_compile.compile(EVAL_PATH, doraise=True)
    print("  ✅ Syntax OK")
except py_compile.PyCompileError as e:
    print(f"  ❌ Syntax error: {e}")

print(f"""
📋 v5.1 HOTFIX Summary:
  1. CSV loading: MERGE horizontally (was CONCAT vertically!)
     - Replicates training_service._load_from_csv() merge logic
     - Merges game/odds/player/situation/target/team/weather on game_id
  2. H2O categoricals: Convert bool columns to string BEFORE H2OFrame
     - Detects Enum columns from model JSON metadata
     - Also detects common boolean indicator patterns  
     - Passes column_types hint to H2OFrame constructor
  3. Removed orphaned 'if passed:' block from _print_summary
  4. Added diagnostic logging (found vs padded features, data quality)

🔄 Next: docker exec -it royaley_api python scripts/evaluate_models.py --verbose
""")