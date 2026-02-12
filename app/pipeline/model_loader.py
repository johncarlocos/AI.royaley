"""
ROYALEY - Model Loader
Loads trained sklearn models + calibrators for live predictions.

Currently uses sklearn models only (70% ensemble weight, proven working).
H2O/autogluon/deep_learning integration can be added later.
"""

import pickle
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Singleton cache for loaded models
_model_cache: Dict[str, dict] = {}
_MODEL_BASE_DIR = Path("/app/models")


def _cache_key(sport_code: str, bet_type: str) -> str:
    return f"{sport_code}_{bet_type}"


def load_model(sport_code: str, bet_type: str) -> Optional[dict]:
    """
    Load sklearn model + scaler + calibrator for a sport/bet_type combo.
    Returns cached version if already loaded.
    
    Returns dict with keys: 'model', 'scaler', 'calibrator' (optional)
    """
    key = _cache_key(sport_code, bet_type)
    if key in _model_cache:
        return _model_cache[key]
    
    model_path = _MODEL_BASE_DIR / "sklearn" / sport_code / bet_type / "model.pkl"
    scaler_path = _MODEL_BASE_DIR / "sklearn" / sport_code / bet_type / "scaler.pkl"
    calibrator_path = _MODEL_BASE_DIR / sport_code / bet_type / "calibrator.pkl"
    
    if not model_path.exists():
        logger.warning(f"No sklearn model found: {model_path}")
        return None
    
    try:
        # Load model
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"Loaded model: {model_path} ({type(model).__name__})")
        
        # Load scaler
        scaler = None
        if scaler_path.exists():
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
            logger.info(f"Loaded scaler: {scaler_path}")
        
        # Load calibrator
        calibrator = None
        if calibrator_path.exists():
            with open(calibrator_path, "rb") as f:
                calibrator = pickle.load(f)
            logger.info(f"Loaded calibrator: {calibrator_path}")
        
        entry = {
            "model": model,
            "scaler": scaler,
            "calibrator": calibrator,
            "n_features": model.n_features_in_ if hasattr(model, "n_features_in_") else None,
        }
        _model_cache[key] = entry
        return entry
        
    except Exception as e:
        logger.error(f"Failed to load model {sport_code}/{bet_type}: {e}")
        return None


def predict_probability(
    sport_code: str,
    bet_type: str,
    features: np.ndarray = None,
    feature_dict: Dict[str, float] = None,
) -> Optional[Tuple[float, float]]:
    """
    Run prediction using loaded sklearn model.
    
    Args:
        sport_code: Sport code (NFL, NBA, etc.)
        bet_type: spread, total, or moneyline
        features: Feature array shape (1, N) in correct column order (deprecated)
        feature_dict: Feature name → value dict (preferred, auto-selects correct features)
    
    Returns:
        Tuple of (home_prob, away_prob) or (over_prob, under_prob)
        Returns None on failure
    """
    entry = load_model(sport_code, bet_type)
    if entry is None:
        return None
    
    model = entry["model"]
    scaler = entry["scaler"]
    calibrator = entry["calibrator"]
    
    try:
        # Build feature array from dict using scaler's expected feature order
        if feature_dict is not None and scaler is not None and hasattr(scaler, "feature_names_in_"):
            import pandas as pd
            expected_features = list(scaler.feature_names_in_)
            feature_values = [float(feature_dict.get(f, 0.0)) for f in expected_features]
            features = pd.DataFrame([feature_values], columns=expected_features)
        elif feature_dict is not None:
            # No scaler with feature names — use the full 87-feature order
            from app.pipeline.live_feature_builder import FEATURE_NAMES_87
            feature_values = [float(feature_dict.get(f, 0.0)) for f in FEATURE_NAMES_87]
            features = np.array([feature_values])
        
        if features is None:
            logger.error(f"No features provided for {sport_code}/{bet_type}")
            return None
        
        # Scale features
        if scaler is not None:
            features = scaler.transform(features)
        
        # Ensure numpy array for predict
        if not isinstance(features, np.ndarray):
            features = np.array(features)
        
        # Predict
        proba = model.predict_proba(features)
        
        # proba is [[prob_class_0, prob_class_1]]
        # class 1 = home win / over / home cover (positive outcome)
        prob_positive = float(proba[0][1])
        prob_negative = float(proba[0][0])
        
        # Apply calibration if available
        if calibrator is not None and hasattr(calibrator, "calibrate"):
            try:
                prob_positive = float(calibrator.calibrate(prob_positive))
                prob_negative = 1.0 - prob_positive
            except Exception as e:
                logger.warning(f"Calibration failed, using raw: {e}")
        
        return (prob_positive, prob_negative)
        
    except Exception as e:
        logger.error(f"Prediction failed for {sport_code}/{bet_type}: {e}")
        return None


def preload_all_models():
    """Pre-load all available models into cache at startup."""
    loaded = 0
    failed = 0
    sklearn_dir = _MODEL_BASE_DIR / "sklearn"
    
    if not sklearn_dir.exists():
        logger.warning(f"sklearn model directory not found: {sklearn_dir}")
        return
    
    for sport_dir in sorted(sklearn_dir.iterdir()):
        if not sport_dir.is_dir():
            continue
        sport_code = sport_dir.name
        for bet_dir in sorted(sport_dir.iterdir()):
            if not bet_dir.is_dir():
                continue
            bet_type = bet_dir.name
            model_file = bet_dir / "model.pkl"
            if model_file.exists():
                result = load_model(sport_code, bet_type)
                if result:
                    loaded += 1
                else:
                    failed += 1
    
    logger.info(f"Model preload complete: {loaded} loaded, {failed} failed, {loaded + failed} total")


def get_available_models() -> Dict[str, list]:
    """Return dict of sport → [bet_types] for available models."""
    available = {}
    sklearn_dir = _MODEL_BASE_DIR / "sklearn"
    
    if not sklearn_dir.exists():
        return available
    
    for sport_dir in sorted(sklearn_dir.iterdir()):
        if not sport_dir.is_dir():
            continue
        sport_code = sport_dir.name
        bet_types = []
        for bet_dir in sorted(sport_dir.iterdir()):
            if bet_dir.is_dir() and (bet_dir / "model.pkl").exists():
                bet_types.append(bet_dir.name)
        if bet_types:
            available[sport_code] = bet_types
    
    return available