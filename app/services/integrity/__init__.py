"""
Prediction Integrity Services for ROYALEY.

Includes SHA-256 prediction lock-in and SHAP explanations.
"""

from .sha256_lockln import (
    PredictionHasher,
    hash_prediction,
    verify_prediction_hash,
    PredictionIntegrity,
)
from .shap_explainer import (
    SHAPExplainer,
    PredictionExplanation,
    FeatureImpact,
    generate_shap_explanation,
)

__all__ = [
    # SHA-256
    "PredictionHasher",
    "hash_prediction",
    "verify_prediction_hash",
    "PredictionIntegrity",
    # SHAP
    "SHAPExplainer",
    "PredictionExplanation",
    "FeatureImpact",
    "generate_shap_explanation",
]
