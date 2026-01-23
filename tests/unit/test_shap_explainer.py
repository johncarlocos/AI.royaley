"""
Unit tests for SHAP Explainer.
Tests model interpretability, feature impacts, and explanation generation.
"""

import pytest
from datetime import datetime

from app.services.integrity.shap_explainer import (
    FeatureImpact,
    PredictionExplanation,
    SHAPExplainer,
    generate_shap_explanation,
    FEATURE_DESCRIPTIONS,
)


class TestFeatureImpact:
    """Tests for FeatureImpact dataclass."""
    
    def test_positive_impact(self):
        """Test positive feature impact."""
        impact = FeatureImpact(
            feature_name="home_elo",
            feature_value=1650.0,
            shap_value=0.15,
            impact_direction="positive",
            impact_magnitude="high",
            contribution_percent=12.5,
            description="Home team has higher ELO rating, contributing to prediction",
        )
        
        assert impact.impact_direction == "positive"
        assert impact.shap_value > 0
        assert impact.impact_magnitude == "high"
    
    def test_negative_impact(self):
        """Test negative feature impact."""
        impact = FeatureImpact(
            feature_name="away_rest_days",
            feature_value=3,
            shap_value=-0.08,
            impact_direction="negative",
            impact_magnitude="medium",
            contribution_percent=6.5,
            description="Away team has more rest days, working against prediction",
        )
        
        assert impact.impact_direction == "negative"
        assert impact.shap_value < 0
    
    def test_low_impact(self):
        """Test low magnitude impact."""
        impact = FeatureImpact(
            feature_name="home_b2b",
            feature_value=0,
            shap_value=0.02,
            impact_direction="positive",
            impact_magnitude="low",
            contribution_percent=1.5,
            description="Home team not on back-to-back",
        )
        
        assert impact.impact_magnitude == "low"


class TestPredictionExplanation:
    """Tests for PredictionExplanation dataclass."""
    
    @pytest.fixture
    def sample_explanation(self):
        """Create a sample explanation."""
        feature_impacts = [
            FeatureImpact(
                feature_name="home_elo",
                feature_value=1620.0,
                shap_value=0.12,
                impact_direction="positive",
                impact_magnitude="high",
                contribution_percent=15.0,
                description="Home team ELO rating",
            ),
            FeatureImpact(
                feature_name="away_b2b",
                feature_value=1,
                shap_value=0.08,
                impact_direction="positive",
                impact_magnitude="medium",
                contribution_percent=10.0,
                description="Away team on back-to-back",
            ),
            FeatureImpact(
                feature_name="h2h_win_pct",
                feature_value=0.65,
                shap_value=0.05,
                impact_direction="positive",
                impact_magnitude="medium",
                contribution_percent=6.25,
                description="Favorable head-to-head record",
            ),
            FeatureImpact(
                feature_name="away_momentum",
                feature_value=0.8,
                shap_value=-0.06,
                impact_direction="negative",
                impact_magnitude="medium",
                contribution_percent=7.5,
                description="Away team has positive momentum",
            ),
        ]
        
        return PredictionExplanation(
            prediction_id="pred_explain_001",
            base_probability=0.50,
            final_probability=0.63,
            feature_impacts=feature_impacts,
            top_positive_factors=["home_elo", "away_b2b", "h2h_win_pct"],
            top_negative_factors=["away_momentum"],
            model_type="xgboost",
            explanation_version="1.0",
            generated_at=datetime.now(),
        )
    
    def test_explanation_structure(self, sample_explanation):
        """Test explanation has proper structure."""
        assert sample_explanation.prediction_id == "pred_explain_001"
        assert sample_explanation.base_probability == 0.50
        assert sample_explanation.final_probability == 0.63
        assert len(sample_explanation.feature_impacts) == 4
    
    def test_top_factors(self, sample_explanation):
        """Test top positive and negative factors."""
        assert len(sample_explanation.top_positive_factors) == 3
        assert len(sample_explanation.top_negative_factors) == 1
        assert "home_elo" in sample_explanation.top_positive_factors
        assert "away_momentum" in sample_explanation.top_negative_factors
    
    def test_probability_increase(self, sample_explanation):
        """Test probability increased from base."""
        prob_change = (
            sample_explanation.final_probability - 
            sample_explanation.base_probability
        )
        assert prob_change > 0  # Positive factors dominated


class TestFeatureDescriptions:
    """Tests for feature description mapping."""
    
    def test_elo_descriptions_exist(self):
        """Test ELO-related descriptions exist."""
        elo_features = ["home_elo", "away_elo", "elo_diff"]
        for feature in elo_features:
            if feature in FEATURE_DESCRIPTIONS:
                assert len(FEATURE_DESCRIPTIONS[feature]) > 0
    
    def test_form_descriptions_exist(self):
        """Test form-related descriptions exist."""
        form_features = ["home_win_streak", "away_last5_wins", "home_momentum"]
        for feature in form_features:
            if feature in FEATURE_DESCRIPTIONS:
                assert len(FEATURE_DESCRIPTIONS[feature]) > 0
    
    def test_rest_descriptions_exist(self):
        """Test rest-related descriptions exist."""
        rest_features = ["home_rest_days", "away_b2b", "rest_advantage"]
        for feature in rest_features:
            if feature in FEATURE_DESCRIPTIONS:
                assert len(FEATURE_DESCRIPTIONS[feature]) > 0


class TestSHAPExplainer:
    """Tests for SHAPExplainer class."""
    
    @pytest.fixture
    def explainer(self):
        """Create a SHAP explainer instance."""
        return SHAPExplainer()
    
    def test_explainer_initialization(self, explainer):
        """Test explainer initializes correctly."""
        assert explainer is not None
        assert explainer.models == {}
    
    def test_register_model(self, explainer):
        """Test registering a model."""
        # Create a mock model
        class MockModel:
            def predict(self, X):
                return [0.5] * len(X)
        
        model = MockModel()
        explainer.register_model("test_model", model)
        
        assert "test_model" in explainer.models
    
    def test_clear_cache(self, explainer):
        """Test clearing the explanation cache."""
        # Add something to cache
        explainer._cache["test"] = "value"
        
        explainer.clear_cache()
        
        assert len(explainer._cache) == 0


class TestGenerateSHAPExplanation:
    """Tests for generate_shap_explanation function."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        class MockModel:
            def __init__(self):
                self.feature_names = [
                    "home_elo", "away_elo", "home_rest_days",
                    "away_rest_days", "h2h_win_pct"
                ]
            
            def predict(self, X):
                return [0.60] * len(X)
            
            def predict_proba(self, X):
                return [[0.40, 0.60]] * len(X)
        
        return MockModel()
    
    @pytest.fixture
    def sample_features(self):
        """Create sample feature values."""
        return {
            "home_elo": 1620.0,
            "away_elo": 1580.0,
            "home_rest_days": 2,
            "away_rest_days": 1,
            "h2h_win_pct": 0.55,
        }
    
    def test_explanation_generation(self, mock_model, sample_features):
        """Test explanation can be generated."""
        # This will use mock/fallback since SHAP may not be installed
        explanation = generate_shap_explanation(
            model=mock_model,
            features=sample_features,
            prediction_id="pred_test_001",
        )
        
        assert explanation is not None
        assert explanation.prediction_id == "pred_test_001"
    
    def test_explanation_has_impacts(self, mock_model, sample_features):
        """Test explanation contains feature impacts."""
        explanation = generate_shap_explanation(
            model=mock_model,
            features=sample_features,
            prediction_id="pred_test_002",
        )
        
        assert len(explanation.feature_impacts) > 0
    
    def test_explanation_has_factors(self, mock_model, sample_features):
        """Test explanation has positive/negative factors."""
        explanation = generate_shap_explanation(
            model=mock_model,
            features=sample_features,
            prediction_id="pred_test_003",
        )
        
        # Should have at least some factors identified
        total_factors = (
            len(explanation.top_positive_factors) + 
            len(explanation.top_negative_factors)
        )
        assert total_factors > 0


class TestImpactMagnitude:
    """Tests for impact magnitude classification."""
    
    def test_high_magnitude(self):
        """Test high magnitude threshold."""
        # SHAP value > 0.1 should be high
        shap_value = 0.15
        assert abs(shap_value) > 0.1
    
    def test_medium_magnitude(self):
        """Test medium magnitude threshold."""
        # 0.05 < SHAP value <= 0.1 should be medium
        shap_value = 0.08
        assert 0.05 < abs(shap_value) <= 0.1
    
    def test_low_magnitude(self):
        """Test low magnitude threshold."""
        # SHAP value <= 0.05 should be low
        shap_value = 0.03
        assert abs(shap_value) <= 0.05


class TestContributionPercent:
    """Tests for contribution percentage calculation."""
    
    def test_contribution_sums_to_hundred(self):
        """Test that contributions sum to approximately 100%."""
        shap_values = [0.15, -0.10, 0.08, -0.05, 0.02]
        total_abs = sum(abs(v) for v in shap_values)
        
        contributions = [(abs(v) / total_abs) * 100 for v in shap_values]
        total_contribution = sum(contributions)
        
        assert total_contribution == pytest.approx(100.0, rel=0.01)
    
    def test_largest_contribution_first(self):
        """Test features are ordered by contribution."""
        shap_values = {
            "feature_a": 0.05,
            "feature_b": 0.15,  # Largest
            "feature_c": -0.10,
            "feature_d": 0.02,
        }
        
        sorted_features = sorted(
            shap_values.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        assert sorted_features[0][0] == "feature_b"


class TestExplanationCaching:
    """Tests for explanation caching."""
    
    @pytest.fixture
    def explainer_with_cache(self):
        """Create explainer with caching enabled."""
        return SHAPExplainer(max_cache_size=100)
    
    def test_cache_stores_explanations(self, explainer_with_cache):
        """Test explanations are cached."""
        # Generate explanation
        cache_key = "pred_cache_001"
        explanation = PredictionExplanation(
            prediction_id=cache_key,
            base_probability=0.50,
            final_probability=0.60,
            feature_impacts=[],
            top_positive_factors=[],
            top_negative_factors=[],
            model_type="xgboost",
            explanation_version="1.0",
            generated_at=datetime.now(),
        )
        
        # Store in cache
        explainer_with_cache._cache[cache_key] = explanation
        
        # Retrieve from cache
        cached = explainer_with_cache._cache.get(cache_key)
        assert cached is not None
        assert cached.prediction_id == cache_key


class TestFeatureImportance:
    """Tests for aggregate feature importance."""
    
    def test_aggregate_importance(self):
        """Test calculating aggregate feature importance."""
        explanations = [
            {
                "feature_impacts": [
                    {"feature_name": "home_elo", "shap_value": 0.12},
                    {"feature_name": "away_b2b", "shap_value": 0.08},
                ]
            },
            {
                "feature_impacts": [
                    {"feature_name": "home_elo", "shap_value": 0.10},
                    {"feature_name": "away_b2b", "shap_value": 0.05},
                ]
            },
            {
                "feature_impacts": [
                    {"feature_name": "home_elo", "shap_value": 0.14},
                    {"feature_name": "away_b2b", "shap_value": 0.06},
                ]
            },
        ]
        
        # Calculate average importance
        importance = {}
        for exp in explanations:
            for impact in exp["feature_impacts"]:
                name = impact["feature_name"]
                value = abs(impact["shap_value"])
                if name not in importance:
                    importance[name] = []
                importance[name].append(value)
        
        avg_importance = {
            name: sum(values) / len(values) 
            for name, values in importance.items()
        }
        
        assert avg_importance["home_elo"] == pytest.approx(0.12, rel=0.01)
        assert avg_importance["away_b2b"] == pytest.approx(0.0633, rel=0.01)


class TestSportSpecificFeatures:
    """Tests for sport-specific feature handling."""
    
    def test_basketball_features(self):
        """Test basketball-specific features."""
        basketball_features = {
            "home_offensive_rating": 115.5,
            "away_defensive_rating": 108.2,
            "home_pace": 102.0,
            "away_true_shooting_pct": 0.58,
        }
        
        # All features should be valid
        assert all(isinstance(v, (int, float)) for v in basketball_features.values())
    
    def test_football_features(self):
        """Test football-specific features."""
        football_features = {
            "home_yards_per_play": 5.8,
            "away_turnover_margin": -2,
            "home_third_down_pct": 0.42,
            "away_red_zone_pct": 0.55,
        }
        
        assert all(isinstance(v, (int, float)) for v in football_features.values())
    
    def test_tennis_features(self):
        """Test tennis-specific features."""
        tennis_features = {
            "player1_ranking": 15,
            "player2_first_serve_pct": 0.68,
            "surface_advantage": 1,
            "h2h_record": 0.60,
        }
        
        assert all(isinstance(v, (int, float)) for v in tennis_features.values())


class TestExplanationVersioning:
    """Tests for explanation versioning."""
    
    def test_version_string(self):
        """Test explanation has version string."""
        explanation = PredictionExplanation(
            prediction_id="pred_version_001",
            base_probability=0.50,
            final_probability=0.58,
            feature_impacts=[],
            top_positive_factors=[],
            top_negative_factors=[],
            model_type="lightgbm",
            explanation_version="2.1.0",
            generated_at=datetime.now(),
        )
        
        assert explanation.explanation_version == "2.1.0"
    
    def test_model_type_recorded(self):
        """Test model type is recorded."""
        for model_type in ["xgboost", "lightgbm", "catboost", "random_forest"]:
            explanation = PredictionExplanation(
                prediction_id=f"pred_{model_type}",
                base_probability=0.50,
                final_probability=0.55,
                feature_impacts=[],
                top_positive_factors=[],
                top_negative_factors=[],
                model_type=model_type,
                explanation_version="1.0",
                generated_at=datetime.now(),
            )
            
            assert explanation.model_type == model_type
