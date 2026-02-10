#!/usr/bin/env python3
"""
ROYALEY - Phase 2: Model Evaluation & Scorecard
Scans all 156 trained models, loads each, evaluates on validation data,
and generates a ranked CSV scorecard.

Usage:
    python scripts/evaluate_models.py --models-dir /nvme0n1-disk/royaley/models
    python scripts/evaluate_models.py --models-dir /nvme0n1-disk/royaley/models --sport NFL
    python scripts/evaluate_models.py --models-dir /nvme0n1-disk/royaley/models --output scorecard.csv
"""

import asyncio
import argparse
import logging
import sys
import pickle
import json
import csv
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

console = Console()
logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# ============================================================================
# CONFIGURATION
# ============================================================================

SPORTS = ["NFL", "NCAAF", "CFL", "NBA", "NCAAB", "WNBA", "NHL", "MLB", "ATP", "WTA"]
BET_TYPES = ["spread", "moneyline", "total"]
FRAMEWORKS = ["h2o", "sklearn", "tensorflow", "autogluon", "quantum"]

# Target columns per bet type
TARGET_COLUMNS = {
    "spread": "spread_result",
    "moneyline": "home_win",
    "total": "over_result",
}

# Minimum acceptable thresholds
MIN_ACCURACY = 0.52
MIN_AUC = 0.52
MAX_CALIBRATION_ERROR = 0.15


@dataclass
class ModelScore:
    """Evaluation result for a single model."""
    sport: str = ""
    bet_type: str = ""
    framework: str = ""
    model_path: str = ""

    # Core metrics
    accuracy: float = 0.0
    auc_roc: float = 0.0
    log_loss: float = 999.0
    brier_score: float = 999.0
    f1_score: float = 0.0

    # Calibration
    ece: float = 999.0  # Expected Calibration Error

    # Betting metrics
    simulated_roi: float = 0.0
    win_rate_tier_a: float = 0.0
    win_rate_tier_b: float = 0.0
    n_tier_a: int = 0
    n_tier_b: int = 0

    # Info
    n_validation_samples: int = 0
    n_features: int = 0
    model_size_mb: float = 0.0
    load_success: bool = False
    error: str = ""

    # Ranking
    composite_score: float = 0.0
    rank: int = 0
    passes_threshold: bool = False


# ============================================================================
# MODEL DISCOVERY
# ============================================================================

def discover_models(models_dir: Path) -> List[Dict[str, str]]:
    """
    Discover all trained model artifacts.

    Directory structure:
        {models_dir}/sklearn/{SPORT}/{bet_type}/model.pkl
        {models_dir}/tensorflow/{SPORT}/{bet_type}/dense_model/
        {models_dir}/h2o/{SPORT}/{bet_type}/*.bin or AutoML folders
        {models_dir}/autogluon/{SPORT}/{bet_type}/
        {models_dir}/quantum/{SPORT}/{bet_type}/
        {models_dir}/{SPORT}/{bet_type}/meta_ensemble.pkl  (ensemble)
    """
    discovered = []

    # Framework-specific models
    for framework in FRAMEWORKS:
        fw_dir = models_dir / framework
        if not fw_dir.exists():
            continue

        for sport_dir in sorted(fw_dir.iterdir()):
            if not sport_dir.is_dir() or sport_dir.name not in SPORTS:
                continue
            sport = sport_dir.name

            for bt_dir in sorted(sport_dir.iterdir()):
                if not bt_dir.is_dir() or bt_dir.name not in BET_TYPES:
                    continue
                bet_type = bt_dir.name

                # Determine model file path
                model_path = _find_model_file(bt_dir, framework)
                if model_path:
                    discovered.append({
                        "sport": sport,
                        "bet_type": bet_type,
                        "framework": framework,
                        "model_path": str(model_path),
                        "model_dir": str(bt_dir),
                    })

    # Meta-ensemble models (at sport level)
    for sport_dir in sorted(models_dir.iterdir()):
        if not sport_dir.is_dir() or sport_dir.name not in SPORTS:
            continue
        sport = sport_dir.name

        for bt_dir in sorted(sport_dir.iterdir()):
            if not bt_dir.is_dir() or bt_dir.name not in BET_TYPES:
                continue
            bet_type = bt_dir.name

            ensemble_path = bt_dir / "meta_ensemble.pkl"
            if ensemble_path.exists():
                discovered.append({
                    "sport": sport,
                    "bet_type": bet_type,
                    "framework": "meta_ensemble",
                    "model_path": str(ensemble_path),
                    "model_dir": str(bt_dir),
                })

    return discovered


def _find_model_file(bt_dir: Path, framework: str) -> Optional[Path]:
    """Find the primary model file in a bet-type directory."""
    if framework == "sklearn":
        p = bt_dir / "model.pkl"
        return p if p.exists() else None

    elif framework == "tensorflow":
        p = bt_dir / "dense_model"
        if p.exists() and p.is_dir():
            return p
        # Also check for .h5 or .keras
        for ext in [".h5", ".keras"]:
            p = list(bt_dir.glob(f"*{ext}"))
            if p:
                return p[0]
        return None

    elif framework == "h2o":
        # H2O saves as directories (AutoML model names) or MOJO
        mojo_dir = bt_dir / "mojo"
        if mojo_dir.exists():
            return mojo_dir
        # Find StackedEnsemble or best model directory
        for d in sorted(bt_dir.iterdir()):
            if d.is_dir() and "StackedEnsemble" in d.name:
                return d
        # Fall back to first model directory (DRF, GBM, etc.)
        for d in sorted(bt_dir.iterdir()):
            if d.is_dir() and d.name not in ["mojo"]:
                return d
        return None

    elif framework == "autogluon":
        # AutoGluon saves entire predictor directory
        if (bt_dir / "predictor.pkl").exists() or (bt_dir / "metadata.json").exists():
            return bt_dir
        # Check for AutogluonModels subdirectory
        ag_dir = bt_dir / "AutogluonModels"
        if ag_dir.exists():
            return ag_dir
        return bt_dir if any(bt_dir.iterdir()) else None

    elif framework == "quantum":
        p = bt_dir / "model.pkl"
        return p if p.exists() else None

    return None


# ============================================================================
# MODEL LOADING & PREDICTION
# ============================================================================

def load_and_predict(model_info: Dict, validation_df: pd.DataFrame,
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

    return None


def _predict_sklearn(model_path: str, model_dir: str,
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

    return np.clip(probs, 0.001, 0.999)


def _predict_tensorflow(model_path: str, model_dir: str,
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

    return np.clip(probs, 0.001, 0.999)


def _predict_h2o(model_path: str, model_dir: str,
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

    return np.clip(result.astype(float), 0.001, 0.999)


def _predict_autogluon(model_path: str, df: pd.DataFrame,
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
        return np.clip(probs.values.ravel(), 0.001, 0.999)


def _predict_quantum(model_path: str, model_dir: str,
                     df: pd.DataFrame, features: List[str]) -> np.ndarray:
    """Load quantum model and predict."""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    scaler_path = Path(model_dir) / "scaler.pkl"
    X = df[features].values

    if scaler_path.exists():
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        X = scaler.transform(X)

    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)

    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X)
        if probs.ndim == 2:
            probs = probs[:, 1]
    elif hasattr(model, 'predict'):
        probs = model.predict(X)
    else:
        raise ValueError("Quantum model has no predict method")

    return np.clip(probs, 0.001, 0.999)


def _predict_meta_ensemble(model_path: str, model_dir: str,
                           df: pd.DataFrame, features: List[str]) -> np.ndarray:
    """Load meta-ensemble and predict."""
    with open(model_path, 'rb') as f:
        ensemble = pickle.load(f)

    if hasattr(ensemble, 'predict_proba'):
        probs = ensemble.predict_proba(df[features].values)
        if probs.ndim == 2:
            probs = probs[:, 1]
    elif hasattr(ensemble, 'predict'):
        probs = ensemble.predict(df[features].values)
    elif isinstance(ensemble, dict) and 'weights' in ensemble:
        # It's a weights dict — need base models to combine
        logger.warning("Meta-ensemble is weights-only, skipping direct prediction")
        return None
    else:
        raise ValueError(f"Unknown meta_ensemble format: {type(ensemble)}")

    return np.clip(probs, 0.001, 0.999)


# ============================================================================
# METRICS COMPUTATION
# ============================================================================

def compute_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray,
                    implied_probs: np.ndarray = None) -> Dict[str, float]:
    """Compute comprehensive evaluation metrics."""
    from sklearn.metrics import (
        accuracy_score, roc_auc_score, log_loss as sk_log_loss,
        brier_score_loss, f1_score as sk_f1_score
    )

    y_pred_class = (y_pred_proba > 0.5).astype(int)

    metrics = {}

    # Core classification metrics
    metrics["accuracy"] = accuracy_score(y_true, y_pred_class)
    try:
        metrics["auc_roc"] = roc_auc_score(y_true, y_pred_proba)
    except ValueError:
        metrics["auc_roc"] = 0.5

    metrics["log_loss"] = sk_log_loss(y_true, y_pred_proba)
    metrics["brier_score"] = brier_score_loss(y_true, y_pred_proba)
    metrics["f1_score"] = sk_f1_score(y_true, y_pred_class, zero_division=0)

    # Expected Calibration Error (ECE)
    metrics["ece"] = _compute_ece(y_true, y_pred_proba, n_bins=10)

    # Simulated ROI with flat betting
    if implied_probs is not None:
        metrics["simulated_roi"] = _simulate_roi(y_true, y_pred_proba, implied_probs)
    else:
        # Use 52.4% implied (standard -110 vig)
        implied = np.full_like(y_pred_proba, 0.524)
        metrics["simulated_roi"] = _simulate_roi(y_true, y_pred_proba, implied)

    # Tier breakdown
    tier_a_mask = y_pred_proba >= 0.65
    tier_b_mask = (y_pred_proba >= 0.60) & (y_pred_proba < 0.65)

    metrics["win_rate_tier_a"] = accuracy_score(y_true[tier_a_mask], y_pred_class[tier_a_mask]) if tier_a_mask.sum() > 0 else 0.0
    metrics["win_rate_tier_b"] = accuracy_score(y_true[tier_b_mask], y_pred_class[tier_b_mask]) if tier_b_mask.sum() > 0 else 0.0
    metrics["n_tier_a"] = int(tier_a_mask.sum())
    metrics["n_tier_b"] = int(tier_b_mask.sum())

    return metrics


def _compute_ece(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(y_true)

    for i in range(n_bins):
        mask = (y_proba >= bin_boundaries[i]) & (y_proba < bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        bin_accuracy = y_true[mask].mean()
        bin_confidence = y_proba[mask].mean()
        bin_weight = mask.sum() / total
        ece += bin_weight * abs(bin_accuracy - bin_confidence)

    return ece


def _simulate_roi(y_true: np.ndarray, y_proba: np.ndarray,
                  implied_probs: np.ndarray, min_edge: float = 0.03) -> float:
    """Simulate flat-bet ROI on predictions with edge > min_edge."""
    edge = y_proba - implied_probs
    bet_mask = edge > min_edge

    if bet_mask.sum() == 0:
        return 0.0

    # Standard -110 payout (risking 1.10 to win 1.00)
    n_bets = bet_mask.sum()
    wins = y_true[bet_mask].sum()
    losses = n_bets - wins

    # Profit: wins * 1.0 - losses * 1.10
    profit = wins * 1.0 - losses * 1.10
    roi = profit / (n_bets * 1.10)

    return roi


def compute_composite_score(metrics: Dict[str, float]) -> float:
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

    return round(score, 6)


# ============================================================================
# VALIDATION DATA LOADING
# ============================================================================

def load_validation_data(sport: str, bet_type: str,
                         csv_dir: str = None) -> Optional[Tuple[pd.DataFrame, List[str], str]]:
    """
    Load validation data for a sport/bet_type.
    Uses the last 20% of data as validation (walk-forward style).
    """
    csv_paths = [
        Path(csv_dir) if csv_dir else None,
        Path(__file__).parent.parent / "app" / "services" / "ml_csv",
        Path(__file__).parent.parent / "ml_csv",
        Path("/nvme0n1-disk/royaley/ml_csv"),
    ]

    df = None
    target_col = TARGET_COLUMNS.get(bet_type, "home_win")

    for csv_path in csv_paths:
        if csv_path is None or not csv_path.exists():
            continue

        sport_dir = csv_path / sport
        if not sport_dir.exists():
            # Try flat structure
            pattern = f"ml_features_{sport}_*.csv"
            files = list(csv_path.glob(pattern))
            if not files:
                continue

            # Load and merge
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
            # Sport subdirectory exists
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

    # Ensure target exists
    if target_col not in df.columns:
        # Try to reconstruct
        if 'home_score' in df.columns and 'away_score' in df.columns:
            margin = df['home_score'] - df['away_score']
            if target_col == 'home_win':
                df['home_win'] = (margin > 0).astype(int)
            elif target_col == 'spread_result':
                # Need spread line
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

    # Identify feature columns (exclude meta/target/identifier columns)
    exclude_cols = {
        'game_id', 'date', 'scheduled_at', 'home_team', 'away_team',
        'home_team_id', 'away_team_id', 'season', 'season_type',
        'home_score', 'away_score', 'home_win', 'spread_result', 'over_result',
        'spread_close', 'total_close', 'home_ml', 'away_ml',
        'home_odds', 'away_odds', 'over_odds', 'under_odds',
        'spread_home_close', 'spread_away_close',
        'Unnamed: 0', 'index',
    }
    feature_columns = [c for c in df.columns
                       if c not in exclude_cols
                       and not c.startswith('_')
                       and df[c].dtype in ['float64', 'float32', 'int64', 'int32']]

    # Take last 20% as validation
    split_idx = int(len(df) * 0.8)
    val_df = df.iloc[split_idx:].copy().reset_index(drop=True)

    # Clean features — fill NaN, cap inf
    for col in feature_columns:
        val_df[col] = pd.to_numeric(val_df[col], errors='coerce')
    val_df[feature_columns] = val_df[feature_columns].fillna(0)
    val_df[feature_columns] = val_df[feature_columns].replace([np.inf, -np.inf], 0)

    return val_df, feature_columns, target_col


# ============================================================================
# MAIN EVALUATION PIPELINE
# ============================================================================

async def evaluate_all_models(
    models_dir: Path,
    csv_dir: str = None,
    sport_filter: str = None,
    framework_filter: str = None,
    output_path: str = None,
    verbose: bool = False,
) -> List[ModelScore]:
    """Run full evaluation pipeline."""

    console.print(Panel(
        "[bold cyan]ROYALEY Phase 2: Model Evaluation & Scorecard[/bold cyan]\n"
        f"Models: {models_dir}\n"
        f"Filter: sport={sport_filter or 'ALL'}, framework={framework_filter or 'ALL'}",
        title="Configuration"
    ))

    # Step 1: Discover models
    console.print("\n[cyan]Step 1: Discovering models...[/cyan]")
    all_models = discover_models(models_dir)

    if sport_filter:
        all_models = [m for m in all_models if m["sport"] == sport_filter.upper()]
    if framework_filter:
        all_models = [m for m in all_models if m["framework"] == framework_filter.lower()]

    console.print(f"  Found [bold]{len(all_models)}[/bold] model artifacts")

    if not all_models:
        console.print("[red]No models found![/red]")
        return []

    # Group by sport/bet_type for summary
    sport_bt_count = {}
    for m in all_models:
        key = f"{m['sport']}/{m['bet_type']}"
        sport_bt_count[key] = sport_bt_count.get(key, 0) + 1

    for key, count in sorted(sport_bt_count.items()):
        console.print(f"  {key}: {count} models")

    # Step 2: Evaluate each model
    console.print("\n[cyan]Step 2: Evaluating models...[/cyan]")
    scores: List[ModelScore] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console
    ) as progress:
        task = progress.add_task("Evaluating", total=len(all_models))

        for model_info in all_models:
            sport = model_info["sport"]
            bet_type = model_info["bet_type"]
            framework = model_info["framework"]

            progress.update(task, description=f"{framework}/{sport}/{bet_type}")

            score = ModelScore(
                sport=sport,
                bet_type=bet_type,
                framework=framework,
                model_path=model_info["model_path"],
            )

            try:
                # Get model file size
                mp = Path(model_info["model_path"])
                if mp.is_file():
                    score.model_size_mb = mp.stat().st_size / (1024 * 1024)
                elif mp.is_dir():
                    total = sum(f.stat().st_size for f in mp.rglob("*") if f.is_file())
                    score.model_size_mb = total / (1024 * 1024)

                # Load validation data
                val_result = load_validation_data(sport, bet_type, csv_dir)
                if val_result is None:
                    score.error = "No validation data found"
                    scores.append(score)
                    progress.advance(task)
                    continue

                val_df, feature_columns, target_col = val_result
                score.n_validation_samples = len(val_df)
                score.n_features = len(feature_columns)

                # Generate predictions
                y_pred_proba = load_and_predict(model_info, val_df, feature_columns)

                if y_pred_proba is None:
                    score.error = "Prediction failed"
                    scores.append(score)
                    progress.advance(task)
                    continue

                score.load_success = True
                y_true = val_df[target_col].values

                # Filter valid rows (both y_true and y_pred must be valid)
                valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred_proba))
                if valid_mask.sum() < 10:
                    score.error = f"Too few valid predictions: {valid_mask.sum()}"
                    scores.append(score)
                    progress.advance(task)
                    continue

                y_true = y_true[valid_mask]
                y_pred_proba = y_pred_proba[valid_mask]
                score.n_validation_samples = len(y_true)

                # Compute metrics
                metrics = compute_metrics(y_true, y_pred_proba)

                score.accuracy = metrics["accuracy"]
                score.auc_roc = metrics["auc_roc"]
                score.log_loss = metrics["log_loss"]
                score.brier_score = metrics["brier_score"]
                score.f1_score = metrics["f1_score"]
                score.ece = metrics["ece"]
                score.simulated_roi = metrics["simulated_roi"]
                score.win_rate_tier_a = metrics["win_rate_tier_a"]
                score.win_rate_tier_b = metrics["win_rate_tier_b"]
                score.n_tier_a = metrics["n_tier_a"]
                score.n_tier_b = metrics["n_tier_b"]

                # Composite score
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
                    )

            except Exception as e:
                score.error = str(e)
                if verbose:
                    console.print(f"  [red]ERROR {framework}/{sport}/{bet_type}: {e}[/red]")
                    traceback.print_exc()

            scores.append(score)
            progress.advance(task)

    # Step 3: Rank models
    console.print("\n[cyan]Step 3: Ranking models...[/cyan]")

    # Sort by composite score descending
    scores.sort(key=lambda s: s.composite_score, reverse=True)
    for i, s in enumerate(scores):
        s.rank = i + 1

    # Step 4: Print Results
    _print_scorecard(scores)

    # Step 5: Save to CSV
    output_path = output_path or "model_scorecard.csv"
    _save_scorecard_csv(scores, output_path)
    console.print(f"\n[green]Scorecard saved to {output_path}[/green]")

    # Step 6: Print Summary
    _print_summary(scores)

    return scores


def _print_scorecard(scores: List[ModelScore]):
    """Print top models table."""
    table = Table(title="Model Scorecard (Top 30)")

    table.add_column("Rank", style="dim", width=4)
    table.add_column("Sport", style="cyan")
    table.add_column("Bet", style="blue")
    table.add_column("Framework", style="magenta")
    table.add_column("Acc", style="yellow")
    table.add_column("AUC", style="yellow")
    table.add_column("ROI", style="green")
    table.add_column("ECE", style="red")
    table.add_column("Tier-A", style="green")
    table.add_column("Score", style="bold white")
    table.add_column("Pass?", style="white")

    for s in scores[:30]:
        if not s.load_success:
            table.add_row(
                str(s.rank), s.sport, s.bet_type, s.framework,
                "-", "-", "-", "-", "-", "-",
                f"[red]ERR: {s.error[:30]}[/red]"
            )
            continue

        roi_str = f"{s.simulated_roi*100:+.1f}%"
        if s.simulated_roi > 0:
            roi_str = f"[green]{roi_str}[/green]"
        else:
            roi_str = f"[red]{roi_str}[/red]"

        tier_a_str = f"{s.win_rate_tier_a*100:.0f}%({s.n_tier_a})" if s.n_tier_a > 0 else "-"

        pass_str = "[green]✓[/green]" if s.passes_threshold else "[red]✗[/red]"

        table.add_row(
            str(s.rank), s.sport, s.bet_type, s.framework,
            f"{s.accuracy:.3f}", f"{s.auc_roc:.3f}",
            roi_str, f"{s.ece:.3f}", tier_a_str,
            f"{s.composite_score:.4f}", pass_str,
        )

    console.print(table)


def _print_summary(scores: List[ModelScore]):
    """Print summary statistics."""
    loaded = [s for s in scores if s.load_success]
    passed = [s for s in loaded if s.passes_threshold]
    failed = [s for s in loaded if not s.passes_threshold]
    errors = [s for s in scores if not s.load_success]

    console.print(Panel(
        f"[bold]Total Models:[/bold] {len(scores)}\n"
        f"[green]Loaded Successfully:[/green] {len(loaded)}\n"
        f"[green]Passed Thresholds:[/green] {len(passed)} (acc≥{MIN_ACCURACY}, auc≥{MIN_AUC}, ece≤{MAX_CALIBRATION_ERROR})\n"
        f"[red]Below Threshold:[/red] {len(failed)}\n"
        f"[red]Load Errors:[/red] {len(errors)}",
        title="Summary"
    ))

    # Best per sport/bet_type
    if passed:
        console.print("\n[cyan]Best Model per Sport/Bet-Type:[/cyan]")
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

        console.print(best_table)


def _save_scorecard_csv(scores: List[ModelScore], path: str):
    """Save scorecard to CSV."""
    rows = []
    for s in scores:
        rows.append({
            "rank": s.rank,
            "sport": s.sport,
            "bet_type": s.bet_type,
            "framework": s.framework,
            "accuracy": round(s.accuracy, 4),
            "auc_roc": round(s.auc_roc, 4),
            "log_loss": round(s.log_loss, 4),
            "brier_score": round(s.brier_score, 4),
            "f1_score": round(s.f1_score, 4),
            "ece": round(s.ece, 4),
            "simulated_roi": round(s.simulated_roi, 4),
            "win_rate_tier_a": round(s.win_rate_tier_a, 4),
            "win_rate_tier_b": round(s.win_rate_tier_b, 4),
            "n_tier_a": s.n_tier_a,
            "n_tier_b": s.n_tier_b,
            "n_validation_samples": s.n_validation_samples,
            "n_features": s.n_features,
            "model_size_mb": round(s.model_size_mb, 2),
            "composite_score": round(s.composite_score, 6),
            "passes_threshold": s.passes_threshold,
            "load_success": s.load_success,
            "model_path": s.model_path,
            "error": s.error,
        })

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


# ============================================================================
# CLI
# ============================================================================

async def main():
    parser = argparse.ArgumentParser(description="ROYALEY Phase 2: Model Evaluation")
    parser.add_argument("--models-dir", "-m", type=str,
                        default="/nvme0n1-disk/royaley/models",
                        help="Root models directory")
    parser.add_argument("--csv-dir", type=str, default=None,
                        help="Directory with ML training CSVs")
    parser.add_argument("--sport", "-s", type=str, default=None,
                        help="Filter by sport (e.g., NFL)")
    parser.add_argument("--framework", "-f", type=str, default=None,
                        help="Filter by framework (e.g., sklearn)")
    parser.add_argument("--output", "-o", type=str, default="model_scorecard.csv",
                        help="Output CSV path")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    await evaluate_all_models(
        models_dir=Path(args.models_dir),
        csv_dir=args.csv_dir,
        sport_filter=args.sport,
        framework_filter=args.framework,
        output_path=args.output,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    asyncio.run(main())
