"""
SHAP Explanations Service for LOYALEY.

Provides model interpretability through SHAP (SHapley Additive
exPlanations) values for prediction explanations.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from uuid import UUID
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FeatureImpact:
    """Impact of a single feature on prediction."""
    feature_name: str
    feature_value: float
    shap_value: float
    impact_direction: str  # "positive" or "negative"
    impact_magnitude: str  # "high", "medium", "low"
    contribution_percent: float
    description: str
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "feature_name": self.feature_name,
            "feature_value": self.feature_value,
            "shap_value": self.shap_value,
            "impact_direction": self.impact_direction,
            "impact_magnitude": self.impact_magnitude,
            "contribution_percent": self.contribution_percent,
            "description": self.description,
        }


@dataclass
class PredictionExplanation:
    """Complete explanation for a prediction."""
    prediction_id: UUID
    base_probability: float
    final_probability: float
    feature_impacts: List[FeatureImpact]
    top_positive_factors: List[FeatureImpact]
    top_negative_factors: List[FeatureImpact]
    model_type: str
    explanation_version: str
    generated_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "prediction_id": str(self.prediction_id),
            "base_probability": self.base_probability,
            "final_probability": self.final_probability,
            "feature_impacts": [f.to_dict() for f in self.feature_impacts],
            "top_positive_factors": [f.to_dict() for f in self.top_positive_factors],
            "top_negative_factors": [f.to_dict() for f in self.top_negative_factors],
            "model_type": self.model_type,
            "explanation_version": self.explanation_version,
            "generated_at": self.generated_at.isoformat(),
            "metadata": self.metadata,
        }


# Feature descriptions for human-readable explanations
FEATURE_DESCRIPTIONS = {
    # ELO and Ratings
    "home_elo": "Home team ELO rating",
    "away_elo": "Away team ELO rating",
    "elo_diff": "ELO rating difference (home - away)",
    "home_offensive_rating": "Home team points per 100 possessions",
    "home_defensive_rating": "Home team points allowed per 100 possessions",
    "away_offensive_rating": "Away team points per 100 possessions",
    "away_defensive_rating": "Away team points allowed per 100 possessions",
    "net_rating_diff": "Net rating difference",
    
    # Form and Momentum
    "home_win_streak": "Home team current win streak",
    "away_win_streak": "Away team current win streak",
    "home_last5_wins": "Home team wins in last 5 games",
    "away_last5_wins": "Away team wins in last 5 games",
    "home_last10_margin": "Home team average margin last 10 games",
    "away_last10_margin": "Away team average margin last 10 games",
    "home_momentum": "Home team momentum score",
    "away_momentum": "Away team momentum score",
    
    # Rest and Travel
    "home_rest_days": "Days since home team's last game",
    "away_rest_days": "Days since away team's last game",
    "rest_advantage": "Rest days advantage (home - away)",
    "home_b2b": "Home team on back-to-back",
    "away_b2b": "Away team on back-to-back",
    "home_travel_miles": "Miles traveled by home team",
    "away_travel_miles": "Miles traveled by away team",
    
    # Head-to-Head
    "h2h_home_wins": "Head-to-head wins for home team",
    "h2h_away_wins": "Head-to-head wins for away team",
    "h2h_home_win_pct": "Head-to-head win percentage for home",
    "h2h_avg_margin": "Average margin in head-to-head games",
    
    # Line Movement
    "opening_spread": "Opening spread line",
    "current_spread": "Current spread line",
    "spread_movement": "Spread line movement",
    "opening_total": "Opening total line",
    "current_total": "Current total line",
    "total_movement": "Total line movement",
    "public_home_pct": "Public betting percentage on home",
    "steam_move": "Sharp money steam move detected",
    "reverse_line_move": "Reverse line movement detected",
    
    # Situational
    "home_home_record": "Home team's home record",
    "away_away_record": "Away team's away record",
    "home_ats_pct": "Home team against the spread percentage",
    "away_ats_pct": "Away team against the spread percentage",
    
    # Weather (outdoor sports)
    "temperature": "Game temperature",
    "wind_speed": "Wind speed",
    "precipitation_prob": "Precipitation probability",
    
    # Tennis specific
    "ranking_diff": "Ranking difference",
    "surface_win_pct_diff": "Surface-specific win percentage difference",
    "recent_form_diff": "Recent form difference (last 10 matches)",
}


def generate_shap_explanation(
    model,
    features: Dict[str, float],
    prediction_id: UUID,
    final_probability: float,
    model_type: str = "xgboost",
    top_n: int = 10,
) -> PredictionExplanation:
    """
    Generate SHAP explanation for a prediction.
    
    Args:
        model: Trained model (XGBoost, LightGBM, etc.)
        features: Feature dictionary
        prediction_id: Prediction ID
        final_probability: Model's predicted probability
        model_type: Type of model
        top_n: Number of top factors to return
        
    Returns:
        PredictionExplanation
    """
    try:
        import shap
        
        # Create SHAP explainer based on model type
        if model_type in ["xgboost", "lightgbm", "catboost"]:
            explainer = shap.TreeExplainer(model)
        else:
            # Fallback to KernelExplainer for other models
            explainer = shap.KernelExplainer(model.predict_proba, np.zeros((1, len(features))))
        
        # Convert features to array
        feature_names = list(features.keys())
        feature_values = np.array([list(features.values())])
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(feature_values)
        
        # Handle multi-class output
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class
        
        shap_values = shap_values[0]  # First sample
        
        # Get base value (expected value)
        if hasattr(explainer, 'expected_value'):
            base_value = explainer.expected_value
            if isinstance(base_value, np.ndarray):
                base_value = base_value[1] if len(base_value) > 1 else base_value[0]
        else:
            base_value = 0.5
        
        # Create feature impacts
        feature_impacts = _create_feature_impacts(
            feature_names, feature_values[0], shap_values
        )
        
        # Sort by absolute SHAP value
        sorted_impacts = sorted(
            feature_impacts,
            key=lambda x: abs(x.shap_value),
            reverse=True
        )
        
        # Get top positive and negative factors
        top_positive = [f for f in sorted_impacts if f.shap_value > 0][:top_n]
        top_negative = [f for f in sorted_impacts if f.shap_value < 0][:top_n]
        
        return PredictionExplanation(
            prediction_id=prediction_id,
            base_probability=float(base_value),
            final_probability=final_probability,
            feature_impacts=sorted_impacts[:top_n * 2],
            top_positive_factors=top_positive,
            top_negative_factors=top_negative,
            model_type=model_type,
            explanation_version="1.0",
            generated_at=datetime.utcnow(),
        )
        
    except ImportError:
        logger.warning("SHAP not installed, using mock explanation")
        return _generate_mock_explanation(features, prediction_id, final_probability)
    except Exception as e:
        logger.error(f"Error generating SHAP explanation: {e}")
        return _generate_mock_explanation(features, prediction_id, final_probability)


def _create_feature_impacts(
    feature_names: List[str],
    feature_values: np.ndarray,
    shap_values: np.ndarray,
) -> List[FeatureImpact]:
    """Create FeatureImpact objects from SHAP values."""
    impacts = []
    
    total_abs_shap = sum(abs(sv) for sv in shap_values)
    
    for name, value, shap_val in zip(feature_names, feature_values, shap_values):
        # Determine impact direction
        direction = "positive" if shap_val > 0 else "negative"
        
        # Determine impact magnitude
        abs_shap = abs(shap_val)
        if abs_shap > 0.1:
            magnitude = "high"
        elif abs_shap > 0.05:
            magnitude = "medium"
        else:
            magnitude = "low"
        
        # Calculate contribution percentage
        contribution = (abs_shap / total_abs_shap * 100) if total_abs_shap > 0 else 0
        
        # Get description
        description = _get_feature_description(name, value, shap_val)
        
        impacts.append(FeatureImpact(
            feature_name=name,
            feature_value=float(value),
            shap_value=float(shap_val),
            impact_direction=direction,
            impact_magnitude=magnitude,
            contribution_percent=round(contribution, 1),
            description=description,
        ))
    
    return impacts


def _get_feature_description(
    feature_name: str,
    feature_value: float,
    shap_value: float,
) -> str:
    """Generate human-readable description for a feature's impact."""
    base_desc = FEATURE_DESCRIPTIONS.get(feature_name, feature_name.replace("_", " ").title())
    
    direction = "increases" if shap_value > 0 else "decreases"
    
    # Format value for display
    if abs(feature_value) > 100:
        value_str = f"{feature_value:.0f}"
    elif abs(feature_value) > 1:
        value_str = f"{feature_value:.1f}"
    else:
        value_str = f"{feature_value:.3f}"
    
    return f"{base_desc} ({value_str}) {direction} win probability"


def _generate_mock_explanation(
    features: Dict[str, float],
    prediction_id: UUID,
    final_probability: float,
) -> PredictionExplanation:
    """Generate mock explanation when SHAP is unavailable."""
    feature_impacts = []
    
    # Create mock impacts based on feature values
    for name, value in list(features.items())[:20]:
        # Generate pseudo-random SHAP values based on feature characteristics
        if "elo" in name.lower() or "rating" in name.lower():
            shap_val = (value - 1500) / 1000 * 0.1  # Normalize around 1500
        elif "streak" in name.lower() or "wins" in name.lower():
            shap_val = value * 0.02
        elif "rest" in name.lower():
            shap_val = (value - 2) * 0.01
        elif "b2b" in name.lower():
            shap_val = -value * 0.05
        else:
            shap_val = (value - 0.5) * 0.05 if 0 <= value <= 1 else value * 0.001
        
        direction = "positive" if shap_val > 0 else "negative"
        magnitude = "high" if abs(shap_val) > 0.05 else "medium" if abs(shap_val) > 0.02 else "low"
        
        feature_impacts.append(FeatureImpact(
            feature_name=name,
            feature_value=float(value),
            shap_value=float(shap_val),
            impact_direction=direction,
            impact_magnitude=magnitude,
            contribution_percent=round(abs(shap_val) * 100, 1),
            description=_get_feature_description(name, value, shap_val),
        ))
    
    # Sort by absolute SHAP value
    sorted_impacts = sorted(feature_impacts, key=lambda x: abs(x.shap_value), reverse=True)
    
    return PredictionExplanation(
        prediction_id=prediction_id,
        base_probability=0.5,
        final_probability=final_probability,
        feature_impacts=sorted_impacts[:10],
        top_positive_factors=[f for f in sorted_impacts if f.shap_value > 0][:5],
        top_negative_factors=[f for f in sorted_impacts if f.shap_value < 0][:5],
        model_type="mock",
        explanation_version="1.0-mock",
        generated_at=datetime.utcnow(),
        metadata={"note": "SHAP library not available, using estimated impacts"},
    )


class SHAPExplainer:
    """
    SHAP Explainer for model predictions.
    
    Provides methods for generating and caching SHAP explanations
    for predictions.
    """
    
    def __init__(
        self,
        cache_enabled: bool = True,
        max_cache_size: int = 1000,
    ):
        """
        Initialize SHAP explainer.
        
        Args:
            cache_enabled: Whether to cache explanations
            max_cache_size: Maximum cache size
        """
        self.cache_enabled = cache_enabled
        self.max_cache_size = max_cache_size
        self._cache: Dict[str, PredictionExplanation] = {}
        self._models: Dict[str, Any] = {}
    
    def register_model(self, model_id: str, model: Any, model_type: str = "xgboost"):
        """
        Register a model for explanations.
        
        Args:
            model_id: Unique model identifier
            model: Trained model object
            model_type: Type of model
        """
        self._models[model_id] = {
            "model": model,
            "type": model_type,
        }
    
    def explain(
        self,
        prediction_id: UUID,
        features: Dict[str, float],
        final_probability: float,
        model_id: Optional[str] = None,
    ) -> PredictionExplanation:
        """
        Generate explanation for a prediction.
        
        Args:
            prediction_id: Prediction ID
            features: Feature dictionary
            final_probability: Predicted probability
            model_id: Model to use for explanation
            
        Returns:
            PredictionExplanation
        """
        # Check cache
        cache_key = str(prediction_id)
        if self.cache_enabled and cache_key in self._cache:
            return self._cache[cache_key]
        
        # Get model if registered
        model = None
        model_type = "mock"
        
        if model_id and model_id in self._models:
            model = self._models[model_id]["model"]
            model_type = self._models[model_id]["type"]
        
        # Generate explanation
        if model:
            explanation = generate_shap_explanation(
                model, features, prediction_id, final_probability, model_type
            )
        else:
            explanation = _generate_mock_explanation(features, prediction_id, final_probability)
        
        # Cache result
        if self.cache_enabled:
            if len(self._cache) >= self.max_cache_size:
                # Remove oldest entry
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
            self._cache[cache_key] = explanation
        
        return explanation
    
    def batch_explain(
        self,
        predictions: List[Dict[str, Any]],
        model_id: Optional[str] = None,
    ) -> List[PredictionExplanation]:
        """
        Generate explanations for multiple predictions.
        
        Args:
            predictions: List of prediction dictionaries
            model_id: Model to use
            
        Returns:
            List of explanations
        """
        explanations = []
        
        for pred in predictions:
            explanation = self.explain(
                prediction_id=pred.get("id") or pred.get("prediction_id"),
                features=pred.get("features", {}),
                final_probability=pred.get("probability", 0.5),
                model_id=model_id,
            )
            explanations.append(explanation)
        
        return explanations
    
    def clear_cache(self):
        """Clear explanation cache."""
        self._cache = {}
    
    def get_feature_importance(
        self,
        explanations: List[PredictionExplanation],
    ) -> Dict[str, float]:
        """
        Calculate aggregate feature importance from multiple explanations.
        
        Args:
            explanations: List of explanations
            
        Returns:
            Dictionary of feature names to average absolute SHAP values
        """
        importance = {}
        counts = {}
        
        for exp in explanations:
            for impact in exp.feature_impacts:
                name = impact.feature_name
                if name not in importance:
                    importance[name] = 0
                    counts[name] = 0
                importance[name] += abs(impact.shap_value)
                counts[name] += 1
        
        # Calculate averages
        return {
            name: importance[name] / counts[name]
            for name in importance
        }


# =============================================================================
# EXPORT ALIASES FOR BACKWARD COMPATIBILITY
# =============================================================================

PredictionExplainer = SHAPExplainer
"""Alias for SHAPExplainer for backward compatibility."""

__all__ = [
    # Main Classes
    "SHAPExplainer",
    "PredictionExplainer",  # Alias
    # Data Classes
    "FeatureImpact",
    "PredictionExplanation",
]
