"""
ROYALEY - Multi-Framework Model Loader
Loads and runs predictions across all 5 frameworks using meta-ensemble weights.

Frameworks: sklearn, h2o, autogluon, deep_learning (tensorflow), quantum (pennylane)
Each sport/bet_type has a meta_ensemble.pkl with weights for combining predictions.

Architecture:
  meta_ensemble.pkl → {weights: {sklearn: 0.7, autogluon: 0.2, ...}, base_model_paths: {...}}
  calibrator.pkl → ProbabilityCalibrator for final output
"""

import pickle
import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Singleton caches ──────────────────────────────────────────────────────────
_ensemble_cache: Dict[str, dict] = {}   # sport_bet → meta_ensemble info
_model_cache: Dict[str, object] = {}    # framework_sport_bet → loaded model
_MODEL_BASE = Path("/app/models")

# ── Framework availability flags ──────────────────────────────────────────────
_H2O_AVAILABLE = False
_H2O_INITIALIZED = False
_AG_AVAILABLE = False
_TF_AVAILABLE = False
_QUANTUM_AVAILABLE = False


def _check_frameworks():
    """Check which ML frameworks are importable."""
    global _H2O_AVAILABLE, _AG_AVAILABLE, _TF_AVAILABLE, _QUANTUM_AVAILABLE
    try:
        import h2o
        _H2O_AVAILABLE = True
    except ImportError:
        pass
    try:
        from autogluon.tabular import TabularPredictor
        _AG_AVAILABLE = True
    except ImportError:
        pass
    try:
        import tensorflow
        _TF_AVAILABLE = True
    except ImportError:
        pass
    try:
        import pennylane
        _QUANTUM_AVAILABLE = True
    except ImportError:
        pass

_check_frameworks()


# =============================================================================
# META-ENSEMBLE LOADER
# =============================================================================

def _cache_key(sport: str, bet_type: str) -> str:
    return f"{sport}_{bet_type}"


def load_ensemble(sport_code: str, bet_type: str) -> Optional[dict]:
    """
    Load meta_ensemble config + calibrator for a sport/bet_type.
    Returns dict with 'weights', 'base_model_paths', 'calibrator'.
    Falls back to sklearn-only if no meta_ensemble exists.
    """
    key = _cache_key(sport_code, bet_type)
    if key in _ensemble_cache:
        return _ensemble_cache[key]

    meta_path = _MODEL_BASE / sport_code / bet_type / "meta_ensemble.pkl"
    if not meta_path.exists():
        # No meta-ensemble → try sklearn-only fallback
        sklearn_path = _MODEL_BASE / "sklearn" / sport_code / bet_type / "model.pkl"
        if sklearn_path.exists():
            entry = {
                "weights": {"sklearn": 1.0},
                "base_model_paths": {"sklearn": str(sklearn_path)},
                "calibrator": None,
            }
            _ensemble_cache[key] = entry
            return entry
        return None

    try:
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        # Load calibrator
        cal_path = _MODEL_BASE / sport_code / bet_type / "calibrator.pkl"
        calibrator = None
        if cal_path.exists():
            with open(cal_path, "rb") as f:
                calibrator = pickle.load(f)

        entry = {
            "weights": meta.get("weights", {}),
            "base_model_paths": meta.get("base_model_paths", {}),
            "calibrator": calibrator,
        }
        _ensemble_cache[key] = entry
        logger.info(f"Loaded ensemble {sport_code}/{bet_type}: weights={entry['weights']}")
        return entry

    except Exception as e:
        logger.error(f"Failed to load ensemble {sport_code}/{bet_type}: {e}")
        return None


# =============================================================================
# SKLEARN
# =============================================================================

def _load_sklearn(sport: str, bet_type: str) -> Optional[dict]:
    """Load sklearn model + scaler."""
    ckey = f"sklearn_{sport}_{bet_type}"
    if ckey in _model_cache:
        return _model_cache[ckey]

    model_path = _MODEL_BASE / "sklearn" / sport / bet_type / "model.pkl"
    scaler_path = _MODEL_BASE / "sklearn" / sport / bet_type / "scaler.pkl"

    if not model_path.exists():
        return None

    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        scaler = None
        if scaler_path.exists():
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)

        entry = {"model": model, "scaler": scaler}
        _model_cache[ckey] = entry
        logger.info(f"Loaded sklearn {sport}/{bet_type}")
        return entry
    except Exception as e:
        logger.error(f"sklearn load failed {sport}/{bet_type}: {e}")
        return None


def _predict_sklearn(sport: str, bet_type: str, feature_dict: Dict[str, float]) -> Optional[float]:
    """Run sklearn prediction. Returns P(positive class)."""
    entry = _load_sklearn(sport, bet_type)
    if entry is None:
        return None

    try:
        model = entry["model"]
        scaler = entry["scaler"]

        if scaler and hasattr(scaler, "feature_names_in_"):
            cols = list(scaler.feature_names_in_)
        elif hasattr(model, "feature_names_in_"):
            cols = list(model.feature_names_in_)
        else:
            return None

        values = [float(feature_dict.get(f, 0.0)) for f in cols]
        df = pd.DataFrame([values], columns=cols)

        if scaler:
            df = pd.DataFrame(scaler.transform(df), columns=cols)

        proba = model.predict_proba(df)
        return float(proba[0][1])

    except Exception as e:
        logger.warning(f"sklearn predict failed {sport}/{bet_type}: {e}")
        return None


# =============================================================================
# H2O
# =============================================================================

def _init_h2o():
    """Initialize H2O cluster once."""
    global _H2O_INITIALIZED
    if _H2O_INITIALIZED:
        return True
    if not _H2O_AVAILABLE:
        return False
    try:
        import h2o
        h2o.init(nthreads=2, max_mem_size="1G", verbose=False)
        _H2O_INITIALIZED = True
        logger.info("H2O cluster initialized")
        return True
    except Exception as e:
        logger.warning(f"H2O init failed: {e}")
        return False


def _load_h2o_model(sport: str, bet_type: str, model_path: str) -> Optional[object]:
    """Load H2O model."""
    if not _init_h2o():
        return None

    ckey = f"h2o_{sport}_{bet_type}"
    if ckey in _model_cache:
        return _model_cache[ckey]

    try:
        import h2o
        full_path = model_path if model_path.startswith("/") else str(_MODEL_BASE.parent / model_path)
        model = h2o.load_model(full_path)
        _model_cache[ckey] = model
        logger.info(f"Loaded h2o {sport}/{bet_type}")
        return model
    except Exception as e:
        logger.warning(f"h2o load failed {sport}/{bet_type}: {e}")
        return None


def _predict_h2o(sport: str, bet_type: str, model_path: str, feature_dict: Dict[str, float]) -> Optional[float]:
    """Run H2O prediction. Returns P(positive class)."""
    model = _load_h2o_model(sport, bet_type, model_path)
    if model is None:
        return None

    try:
        import h2o

        # Get features expected by model
        cols = model._model_json["output"]["names"][:-1]  # exclude response column
        values = [float(feature_dict.get(f, 0.0)) for f in cols]
        hf = h2o.H2OFrame([values], column_names=cols)

        preds = model.predict(hf)
        # H2O binary classification: [predict, p0, p1]
        p1 = float(preds[0, "p1"])
        return p1

    except Exception as e:
        logger.warning(f"h2o predict failed {sport}/{bet_type}: {e}")
        return None


# =============================================================================
# AUTOGLUON
# =============================================================================

def _load_autogluon(sport: str, bet_type: str, model_path: str) -> Optional[object]:
    """Load AutoGluon predictor."""
    if not _AG_AVAILABLE:
        return None

    ckey = f"ag_{sport}_{bet_type}"
    if ckey in _model_cache:
        return _model_cache[ckey]

    try:
        from autogluon.tabular import TabularPredictor
        full_path = model_path if model_path.startswith("/") else str(_MODEL_BASE.parent / model_path)
        predictor = TabularPredictor.load(full_path, verbosity=0)
        _model_cache[ckey] = predictor
        logger.info(f"Loaded autogluon {sport}/{bet_type}")
        return predictor
    except Exception as e:
        logger.warning(f"autogluon load failed {sport}/{bet_type}: {e}")
        return None


def _predict_autogluon(sport: str, bet_type: str, model_path: str, feature_dict: Dict[str, float]) -> Optional[float]:
    """Run AutoGluon prediction. Returns P(positive class)."""
    predictor = _load_autogluon(sport, bet_type, model_path)
    if predictor is None:
        return None

    try:
        # Get expected features
        if hasattr(predictor, 'feature_metadata') and predictor.feature_metadata is not None:
            features = predictor.feature_metadata.get_features()
        else:
            features = list(feature_dict.keys())

        values = {f: float(feature_dict.get(f, 0.0)) for f in features}
        df = pd.DataFrame([values])

        proba = predictor.predict_proba(df)
        # Returns DataFrame with class columns
        if hasattr(proba, 'columns') and 1 in proba.columns:
            return float(proba.iloc[0][1])
        elif hasattr(proba, 'iloc'):
            return float(proba.iloc[0, -1])
        else:
            return float(proba[0])

    except Exception as e:
        logger.warning(f"autogluon predict failed {sport}/{bet_type}: {e}")
        return None


# =============================================================================
# TENSORFLOW / DEEP LEARNING
# =============================================================================

def _load_tensorflow(sport: str, bet_type: str, model_path: str) -> Optional[object]:
    """Load TensorFlow/Keras dense model."""
    if not _TF_AVAILABLE:
        return None

    ckey = f"tf_{sport}_{bet_type}"
    if ckey in _model_cache:
        return _model_cache[ckey]

    try:
        import tensorflow as tf
        full_path = model_path if model_path.startswith("/") else str(_MODEL_BASE.parent / model_path)
        model = tf.keras.models.load_model(full_path, compile=False)
        _model_cache[ckey] = model
        logger.info(f"Loaded tensorflow {sport}/{bet_type}")
        return model
    except Exception as e:
        logger.warning(f"tensorflow load failed {sport}/{bet_type}: {e}")
        return None


def _predict_tensorflow(sport: str, bet_type: str, model_path: str, feature_dict: Dict[str, float],
                        scaler_features: List[str] = None) -> Optional[float]:
    """Run TensorFlow prediction. Returns P(positive class)."""
    model = _load_tensorflow(sport, bet_type, model_path)
    if model is None:
        return None

    try:
        # Determine expected input size
        input_shape = model.input_shape
        n_features = input_shape[-1] if isinstance(input_shape, tuple) else input_shape[0][-1]

        # Use scaler features for column ordering (matches training)
        if scaler_features:
            cols = scaler_features[:n_features]
        else:
            cols = sorted(feature_dict.keys())[:n_features]

        values = np.array([[float(feature_dict.get(f, 0.0)) for f in cols]], dtype=np.float32)

        # Pad/truncate to expected size
        if values.shape[1] < n_features:
            values = np.hstack([values, np.zeros((1, n_features - values.shape[1]), dtype=np.float32)])
        elif values.shape[1] > n_features:
            values = values[:, :n_features]

        # Normalize (dense models expect standardized input)
        mean = values.mean()
        std = values.std() + 1e-8
        values = (values - mean) / std

        pred = model.predict(values, verbose=0).flatten()
        # Sigmoid output → probability
        return float(np.clip(pred[0], 0.01, 0.99))

    except Exception as e:
        logger.warning(f"tensorflow predict failed {sport}/{bet_type}: {e}")
        return None


# =============================================================================
# QUANTUM (PennyLane)
# =============================================================================

def _load_quantum(sport: str, bet_type: str, model_path: str) -> Optional[object]:
    """Load PennyLane quantum model."""
    if not _QUANTUM_AVAILABLE:
        return None

    ckey = f"q_{sport}_{bet_type}"
    if ckey in _model_cache:
        return _model_cache[ckey]

    try:
        full_path = model_path if model_path.startswith("/") else str(_MODEL_BASE.parent / model_path)
        with open(full_path, "rb") as f:
            config = pickle.load(f)

        from app.services.ml.quantum_ml import PennyLaneQNN
        qnn = PennyLaneQNN(n_qubits=config["n_qubits"], n_layers=config["n_layers"])
        qnn.weights = np.array(config["weights"])
        _model_cache[ckey] = qnn
        logger.info(f"Loaded quantum {sport}/{bet_type}")
        return qnn
    except Exception as e:
        logger.warning(f"quantum load failed {sport}/{bet_type}: {e}")
        return None


def _predict_quantum(sport: str, bet_type: str, model_path: str, feature_dict: Dict[str, float]) -> Optional[float]:
    """Run quantum prediction. Returns P(positive class)."""
    qnn = _load_quantum(sport, bet_type, model_path)
    if qnn is None:
        return None

    try:
        n_qubits = qnn.n_qubits
        cols = sorted(feature_dict.keys())[:n_qubits]
        values = np.array([[float(feature_dict.get(f, 0.0)) for f in cols]])

        if hasattr(qnn, "predict_proba"):
            proba = qnn.predict_proba(values)
            if proba.ndim > 1:
                return float(proba[0, 1])
            return float(proba[0])
        else:
            pred = qnn.predict(values)
            return 0.7 if pred[0] > 0.5 else 0.3

    except Exception as e:
        logger.warning(f"quantum predict failed {sport}/{bet_type}: {e}")
        return None


# =============================================================================
# MAIN ENSEMBLE PREDICT
# =============================================================================

def predict_probability(
    sport_code: str,
    bet_type: str,
    feature_dict: Dict[str, float] = None,
    **kwargs,
) -> Optional[Tuple[float, float]]:
    """
    Run weighted ensemble prediction across all available frameworks.

    Uses meta_ensemble.pkl weights to combine predictions:
      WTA:  autogluon(0.9) + deep_learning(0.1)
      ATP:  deep_learning(0.6) + h2o(0.2) + autogluon(0.1) + sklearn(0.1)
      NFL:  sklearn(0.7) + autogluon(0.2) + deep_learning(0.05) + h2o(0.05)
      etc.

    Returns:
        (P(positive), P(negative)) where positive = home_cover/over/home_win
        None if no frameworks produce predictions
    """
    ensemble = load_ensemble(sport_code, bet_type)
    if ensemble is None or feature_dict is None:
        return None

    weights = ensemble["weights"]
    paths = ensemble["base_model_paths"]
    calibrator = ensemble["calibrator"]

    # Get sklearn scaler features for TF model column ordering
    sklearn_entry = _load_sklearn(sport_code, bet_type)
    scaler_features = None
    if sklearn_entry and sklearn_entry.get("scaler") and hasattr(sklearn_entry["scaler"], "feature_names_in_"):
        scaler_features = list(sklearn_entry["scaler"].feature_names_in_)

    # ── Collect predictions from each framework ──
    framework_probs: Dict[str, float] = {}

    # sklearn
    w_sk = weights.get("sklearn", 0)
    if w_sk > 0 and "sklearn" in paths:
        prob = _predict_sklearn(sport_code, bet_type, feature_dict)
        if prob is not None:
            framework_probs["sklearn"] = prob

    # h2o
    w_h2o = weights.get("h2o", 0)
    if w_h2o > 0 and "h2o" in paths:
        prob = _predict_h2o(sport_code, bet_type, paths["h2o"], feature_dict)
        if prob is not None:
            framework_probs["h2o"] = prob

    # autogluon
    w_ag = weights.get("autogluon", 0)
    if w_ag > 0 and "autogluon" in paths:
        prob = _predict_autogluon(sport_code, bet_type, paths["autogluon"], feature_dict)
        if prob is not None:
            framework_probs["autogluon"] = prob

    # deep_learning (tensorflow)
    w_dl = weights.get("deep_learning", 0)
    if w_dl > 0 and "deep_learning" in paths:
        prob = _predict_tensorflow(sport_code, bet_type, paths["deep_learning"], feature_dict, scaler_features)
        if prob is not None:
            framework_probs["deep_learning"] = prob

    # quantum
    w_q = weights.get("quantum", 0)
    if w_q > 0 and "quantum" in paths:
        prob = _predict_quantum(sport_code, bet_type, paths["quantum"], feature_dict)
        if prob is not None:
            framework_probs["quantum"] = prob

    if not framework_probs:
        logger.warning(f"No frameworks produced predictions for {sport_code}/{bet_type}")
        return None

    # ── Weighted average ──
    total_weight = 0.0
    weighted_sum = 0.0
    for fw, prob in framework_probs.items():
        w = weights.get(fw, 0)
        weighted_sum += prob * w
        total_weight += w

    if total_weight <= 0:
        prob_positive = sum(framework_probs.values()) / len(framework_probs)
    else:
        prob_positive = weighted_sum / total_weight

    prob_positive = float(np.clip(prob_positive, 0.01, 0.99))

    # ── Apply calibration ──
    if calibrator is not None and hasattr(calibrator, "calibrate"):
        try:
            prob_positive = float(calibrator.calibrate(prob_positive))
            prob_positive = float(np.clip(prob_positive, 0.01, 0.99))
        except Exception as e:
            logger.warning(f"Calibration failed: {e}")

    prob_negative = 1.0 - prob_positive

    fw_str = ", ".join(f"{fw}={framework_probs[fw]:.3f}(w={weights.get(fw, 0):.2f})" for fw in framework_probs)
    logger.info(f"  🎯 {sport_code}/{bet_type}: P={prob_positive:.3f} [{fw_str}]")

    return (prob_positive, prob_negative)


def load_model(sport_code: str, bet_type: str) -> Optional[dict]:
    """
    Backward-compatible check: returns truthy if ANY model exists for sport/bet_type.
    Used by fetch_games.py to decide whether to attempt ML prediction.
    """
    return load_ensemble(sport_code, bet_type)


def preload_all_models():
    """Pre-load all available ensemble configs at startup."""
    loaded = 0
    for sport_dir in sorted(_MODEL_BASE.iterdir()):
        if not sport_dir.is_dir():
            continue
        # Skip framework dirs
        if sport_dir.name in ("sklearn", "h2o", "autogluon", "deep_learning", "quantum", "tensorflow"):
            continue
        sport_code = sport_dir.name
        for bet_dir in sorted(sport_dir.iterdir()):
            if not bet_dir.is_dir():
                continue
            bet_type = bet_dir.name
            if (bet_dir / "meta_ensemble.pkl").exists():
                result = load_ensemble(sport_code, bet_type)
                if result:
                    loaded += 1
    logger.info(f"Ensemble preload: {loaded} configs loaded")


def get_available_models() -> Dict[str, list]:
    """Return dict of sport → [bet_types] with available models."""
    available = {}
    for sport_dir in sorted(_MODEL_BASE.iterdir()):
        if not sport_dir.is_dir():
            continue
        if sport_dir.name in ("sklearn", "h2o", "autogluon", "deep_learning", "quantum", "tensorflow"):
            continue
        sport_code = sport_dir.name
        bet_types = []
        for bet_dir in sorted(sport_dir.iterdir()):
            if bet_dir.is_dir() and (bet_dir / "meta_ensemble.pkl").exists():
                bet_types.append(bet_dir.name)
        if bet_types:
            available[sport_code] = bet_types

    # Also include sklearn-only models
    sklearn_dir = _MODEL_BASE / "sklearn"
    if sklearn_dir.exists():
        for sport_dir in sorted(sklearn_dir.iterdir()):
            if not sport_dir.is_dir():
                continue
            sport_code = sport_dir.name
            if sport_code not in available:
                available[sport_code] = []
            for bet_dir in sorted(sport_dir.iterdir()):
                if bet_dir.is_dir() and (bet_dir / "model.pkl").exists():
                    bt = bet_dir.name
                    if bt not in available.get(sport_code, []):
                        available.setdefault(sport_code, []).append(bt)

    return available