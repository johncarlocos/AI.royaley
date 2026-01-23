"""
ROYALEY - SHAP Explainer
Phase 2: Model interpretability with SHAP values

Provides explanations for predictions by calculating feature
importance using SHAP (SHapley Additive exPlanations) values.
"""

import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class FeatureContribution:
    """Single feature's contribution to a prediction"""
    feature_name: str
    feature_value: float
    shap_value: float
    impact: str  # 'positive' or 'negative'
    contribution_pct: float = 0.0  # Percentage of total contribution
    
    def to_dict(self) -> Dict:
        return {
            'feature': self.feature_name,
            'value': self.feature_value,
            'shap_value': self.shap_value,
            'impact': self.impact,
            'contribution_pct': self.contribution_pct,
        }


@dataclass
class PredictionExplanation:
    """Complete explanation for a single prediction"""
    prediction_id: str
    game_id: str
    predicted_probability: float
    base_value: float  # Expected value (average prediction)
    
    # Top contributing features
    top_positive_factors: List[FeatureContribution] = field(default_factory=list)
    top_negative_factors: List[FeatureContribution] = field(default_factory=list)
    all_contributions: List[FeatureContribution] = field(default_factory=list)
    
    # Summary
    total_positive_contribution: float = 0.0
    total_negative_contribution: float = 0.0
    
    # Metadata
    sport_code: str = ""
    bet_type: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict:
        return {
            'prediction_id': self.prediction_id,
            'game_id': self.game_id,
            'predicted_probability': self.predicted_probability,
            'base_value': self.base_value,
            'top_positive_factors': [f.to_dict() for f in self.top_positive_factors],
            'top_negative_factors': [f.to_dict() for f in self.top_negative_factors],
            'total_positive_contribution': self.total_positive_contribution,
            'total_negative_contribution': self.total_negative_contribution,
        }
    
    def get_summary_text(self) -> str:
        """Generate human-readable explanation summary"""
        lines = [
            f"Prediction: {self.predicted_probability:.1%} probability",
            f"Base expectation: {self.base_value:.1%}",
            "",
            "Key factors increasing probability:",
        ]
        
        for factor in self.top_positive_factors[:5]:
            lines.append(f"  • {factor.feature_name}: +{factor.shap_value:.3f}")
        
        lines.append("")
        lines.append("Key factors decreasing probability:")
        
        for factor in self.top_negative_factors[:5]:
            lines.append(f"  • {factor.feature_name}: {factor.shap_value:.3f}")
        
        return "\n".join(lines)


class SHAPExplainer:
    """
    SHAP-based model explainer for sports predictions.
    
    Provides feature importance explanations by computing
    SHAP values for individual predictions.
    """
    
    def __init__(
        self,
        model: Any = None,
        feature_columns: List[str] = None,
        background_data: pd.DataFrame = None,
        n_background_samples: int = 100,
    ):
        """
        Initialize SHAP explainer.
        
        Args:
            model: Trained model (sklearn-compatible or tree model)
            feature_columns: List of feature column names
            background_data: Background data for SHAP calculations
            n_background_samples: Number of background samples to use
        """
        self.model = model
        self.feature_columns = feature_columns or []
        self.background_data = background_data
        self.n_background_samples = n_background_samples
        
        self._explainer = None
        self._expected_value = None
    
    def initialize(
        self,
        model: Any,
        feature_columns: List[str],
        background_data: pd.DataFrame = None,
    ) -> None:
        """
        Initialize the explainer with a model.
        
        Args:
            model: Trained model
            feature_columns: Feature column names
            background_data: Optional background data for SHAP
        """
        self.model = model
        self.feature_columns = feature_columns
        
        if background_data is not None:
            self.background_data = background_data
        
        self._create_explainer()
    
    def _create_explainer(self) -> None:
        """Create the appropriate SHAP explainer based on model type"""
        try:
            import shap
            
            # Determine model type and create appropriate explainer
            model_type = type(self.model).__name__
            
            # Prepare background data
            if self.background_data is not None:
                bg_data = self.background_data[self.feature_columns].copy()
                if len(bg_data) > self.n_background_samples:
                    bg_data = bg_data.sample(
                        n=self.n_background_samples,
                        random_state=42
                    )
            else:
                bg_data = None
            
            # Try TreeExplainer first (fastest for tree models)
            if self._is_tree_model():
                logger.info(f"Creating TreeExplainer for {model_type}")
                self._explainer = shap.TreeExplainer(self.model)
                self._expected_value = self._explainer.expected_value
                
            # Fall back to KernelExplainer for other models
            elif bg_data is not None:
                logger.info(f"Creating KernelExplainer for {model_type}")
                
                # Use predict_proba if available
                if hasattr(self.model, 'predict_proba'):
                    predict_fn = lambda x: self.model.predict_proba(x)[:, 1]
                else:
                    predict_fn = self.model.predict
                
                self._explainer = shap.KernelExplainer(predict_fn, bg_data)
                self._expected_value = self._explainer.expected_value
                
            else:
                logger.warning("No background data provided for non-tree model")
                self._explainer = None
                
        except ImportError:
            logger.warning("SHAP not installed. Install with: pip install shap")
            self._explainer = None
        except Exception as e:
            logger.error(f"Failed to create SHAP explainer: {e}")
            self._explainer = None
    
    def _is_tree_model(self) -> bool:
        """Check if model is a tree-based model"""
        tree_model_names = [
            'XGBClassifier', 'XGBRegressor',
            'LGBMClassifier', 'LGBMRegressor',
            'CatBoostClassifier', 'CatBoostRegressor',
            'RandomForestClassifier', 'RandomForestRegressor',
            'GradientBoostingClassifier', 'GradientBoostingRegressor',
            'ExtraTreesClassifier', 'ExtraTreesRegressor',
            'DecisionTreeClassifier', 'DecisionTreeRegressor',
        ]
        
        model_name = type(self.model).__name__
        return model_name in tree_model_names
    
    def explain_prediction(
        self,
        features: Union[pd.DataFrame, pd.Series, Dict],
        prediction_id: str = "",
        game_id: str = "",
        sport_code: str = "",
        bet_type: str = "",
        n_top_features: int = 10,
    ) -> PredictionExplanation:
        """
        Generate explanation for a single prediction.
        
        Args:
            features: Feature values for the prediction
            prediction_id: Unique prediction identifier
            game_id: Game identifier
            sport_code: Sport code
            bet_type: Bet type
            n_top_features: Number of top features to return
            
        Returns:
            PredictionExplanation with SHAP-based explanations
        """
        # Convert to DataFrame if needed
        if isinstance(features, dict):
            features_df = pd.DataFrame([features])[self.feature_columns]
        elif isinstance(features, pd.Series):
            features_df = pd.DataFrame([features])[self.feature_columns]
        else:
            features_df = features[self.feature_columns].copy()
        
        # Get prediction probability
        if hasattr(self.model, 'predict_proba'):
            prob = self.model.predict_proba(features_df)[:, 1][0]
        else:
            prob = self.model.predict(features_df)[0]
        
        # Calculate SHAP values
        if self._explainer is None:
            # Return basic explanation without SHAP
            return self._create_basic_explanation(
                features_df, prob, prediction_id, game_id,
                sport_code, bet_type
            )
        
        try:
            shap_values = self._explainer.shap_values(features_df)
            
            # Handle different SHAP value formats
            if isinstance(shap_values, list):
                # For binary classification, use positive class
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            
            if len(shap_values.shape) > 1:
                shap_values = shap_values[0]
            
            # Get expected value
            expected = self._expected_value
            if isinstance(expected, (list, np.ndarray)):
                expected = expected[1] if len(expected) > 1 else expected[0]
            
            # Create feature contributions
            contributions = []
            feature_values = features_df.iloc[0].values
            
            for i, (name, val, shap_val) in enumerate(zip(
                self.feature_columns, feature_values, shap_values
            )):
                contributions.append(FeatureContribution(
                    feature_name=name,
                    feature_value=float(val),
                    shap_value=float(shap_val),
                    impact='positive' if shap_val > 0 else 'negative',
                ))
            
            # Calculate contribution percentages
            total_abs = sum(abs(c.shap_value) for c in contributions)
            if total_abs > 0:
                for c in contributions:
                    c.contribution_pct = abs(c.shap_value) / total_abs * 100
            
            # Sort by absolute SHAP value
            contributions.sort(key=lambda x: abs(x.shap_value), reverse=True)
            
            # Separate positive and negative
            positive = [c for c in contributions if c.shap_value > 0]
            negative = [c for c in contributions if c.shap_value < 0]
            
            # Sort negative by most negative first
            negative.sort(key=lambda x: x.shap_value)
            
            return PredictionExplanation(
                prediction_id=prediction_id,
                game_id=game_id,
                predicted_probability=float(prob),
                base_value=float(expected),
                top_positive_factors=positive[:n_top_features],
                top_negative_factors=negative[:n_top_features],
                all_contributions=contributions,
                total_positive_contribution=sum(c.shap_value for c in positive),
                total_negative_contribution=sum(c.shap_value for c in negative),
                sport_code=sport_code,
                bet_type=bet_type,
            )
            
        except Exception as e:
            logger.error(f"SHAP calculation failed: {e}")
            return self._create_basic_explanation(
                features_df, prob, prediction_id, game_id,
                sport_code, bet_type
            )
    
    def explain_batch(
        self,
        features_df: pd.DataFrame,
        prediction_ids: List[str] = None,
        game_ids: List[str] = None,
        sport_code: str = "",
        bet_type: str = "",
        n_top_features: int = 10,
    ) -> List[PredictionExplanation]:
        """
        Generate explanations for multiple predictions.
        
        Args:
            features_df: DataFrame with features for multiple predictions
            prediction_ids: List of prediction identifiers
            game_ids: List of game identifiers
            sport_code: Sport code
            bet_type: Bet type
            n_top_features: Number of top features to return
            
        Returns:
            List of PredictionExplanation objects
        """
        n_samples = len(features_df)
        
        if prediction_ids is None:
            prediction_ids = [f"pred_{i}" for i in range(n_samples)]
        if game_ids is None:
            game_ids = [f"game_{i}" for i in range(n_samples)]
        
        explanations = []
        
        for i in range(n_samples):
            row = features_df.iloc[[i]]
            
            explanation = self.explain_prediction(
                features=row,
                prediction_id=prediction_ids[i],
                game_id=game_ids[i],
                sport_code=sport_code,
                bet_type=bet_type,
                n_top_features=n_top_features,
            )
            
            explanations.append(explanation)
        
        return explanations
    
    def _create_basic_explanation(
        self,
        features_df: pd.DataFrame,
        probability: float,
        prediction_id: str,
        game_id: str,
        sport_code: str,
        bet_type: str,
    ) -> PredictionExplanation:
        """
        Create basic explanation when SHAP is unavailable.
        
        Uses feature values as proxy for importance.
        """
        # Use feature values scaled by their deviation from 0
        contributions = []
        feature_values = features_df.iloc[0].values
        
        for name, val in zip(self.feature_columns, feature_values):
            # Estimate impact based on value magnitude
            estimated_impact = val * 0.01 if val != 0 else 0
            
            contributions.append(FeatureContribution(
                feature_name=name,
                feature_value=float(val),
                shap_value=float(estimated_impact),
                impact='positive' if estimated_impact > 0 else 'negative',
            ))
        
        # Sort by absolute estimated impact
        contributions.sort(key=lambda x: abs(x.shap_value), reverse=True)
        
        positive = [c for c in contributions if c.shap_value > 0]
        negative = [c for c in contributions if c.shap_value < 0]
        
        return PredictionExplanation(
            prediction_id=prediction_id,
            game_id=game_id,
            predicted_probability=probability,
            base_value=0.5,  # Default base value
            top_positive_factors=positive[:10],
            top_negative_factors=negative[:10],
            all_contributions=contributions,
            sport_code=sport_code,
            bet_type=bet_type,
        )
    
    def get_feature_importance(
        self,
        data: pd.DataFrame = None,
        n_samples: int = 100,
    ) -> Dict[str, float]:
        """
        Calculate global feature importance using SHAP.
        
        Args:
            data: Data to calculate importance on
            n_samples: Number of samples to use
            
        Returns:
            Dictionary of feature -> importance
        """
        if self._explainer is None:
            logger.warning("Explainer not initialized")
            return {}
        
        if data is None:
            if self.background_data is not None:
                data = self.background_data
            else:
                logger.warning("No data provided for feature importance")
                return {}
        
        # Sample data
        features_df = data[self.feature_columns].copy()
        if len(features_df) > n_samples:
            features_df = features_df.sample(n=n_samples, random_state=42)
        
        try:
            # Calculate SHAP values
            shap_values = self._explainer.shap_values(features_df)
            
            # Handle different formats
            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            
            # Calculate mean absolute SHAP value for each feature
            importance = np.abs(shap_values).mean(axis=0)
            
            # Create dictionary
            importance_dict = dict(zip(self.feature_columns, importance))
            
            # Sort by importance
            importance_dict = dict(sorted(
                importance_dict.items(),
                key=lambda x: x[1],
                reverse=True
            ))
            
            return importance_dict
            
        except Exception as e:
            logger.error(f"Failed to calculate feature importance: {e}")
            return {}
    
    def save_explanations(
        self,
        explanations: List[PredictionExplanation],
        filepath: str,
    ) -> None:
        """Save explanations to JSON file"""
        import json
        
        data = [exp.to_dict() for exp in explanations]
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Saved {len(explanations)} explanations to {filepath}")


class SHAPExplainerMock:
    """Mock SHAP explainer for testing"""
    
    def __init__(self, *args, **kwargs):
        self.feature_columns = kwargs.get('feature_columns', [])
    
    def initialize(self, model, feature_columns, background_data=None):
        self.feature_columns = feature_columns
    
    def explain_prediction(
        self,
        features,
        prediction_id: str = "",
        game_id: str = "",
        **kwargs
    ) -> PredictionExplanation:
        """Return mock explanation"""
        n_features = len(self.feature_columns)
        
        mock_contributions = []
        for i, name in enumerate(self.feature_columns[:10]):
            shap_val = np.random.uniform(-0.1, 0.1)
            mock_contributions.append(FeatureContribution(
                feature_name=name,
                feature_value=np.random.random(),
                shap_value=shap_val,
                impact='positive' if shap_val > 0 else 'negative',
            ))
        
        positive = [c for c in mock_contributions if c.shap_value > 0]
        negative = [c for c in mock_contributions if c.shap_value < 0]
        
        return PredictionExplanation(
            prediction_id=prediction_id,
            game_id=game_id,
            predicted_probability=np.random.uniform(0.4, 0.7),
            base_value=0.5,
            top_positive_factors=positive,
            top_negative_factors=negative,
            all_contributions=mock_contributions,
        )


def get_shap_explainer(use_mock: bool = False) -> Union[SHAPExplainer, SHAPExplainerMock]:
    """Factory function to get SHAP explainer"""
    if use_mock:
        return SHAPExplainerMock()
    
    try:
        import shap
        return SHAPExplainer()
    except ImportError:
        logger.warning("SHAP not installed, using mock explainer")
        return SHAPExplainerMock()
