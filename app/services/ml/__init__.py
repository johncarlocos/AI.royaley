"""
ROYALEY - Machine Learning Services
Phase 2: ML Pipeline and Advanced Predictions

This module contains all ML-related services including:
- Configuration (MLConfig, SportConfig)
- Feature Engineering (1,380+ features across 10 sports)
- ELO Rating System
- Model Training:
  * H2O AutoML (50+ algorithms, MOJO export)
  * AutoGluon (Multi-layer stack ensembling)
  * Sklearn Ensemble (XGBoost, LightGBM, CatBoost, Random Forest)
  * TensorFlow/LSTM (Deep Learning, time-series)
  * Quantum ML (PennyLane, Qiskit, D-Wave)
- Probability Calibration (Isotonic, Platt, Temperature Scaling)
- Walk-Forward Validation
- Meta-Ensemble Framework
- Model Registry & Versioning
- Advanced Prediction Engine
- Player Props Engine
- SHAP Explanations
- Signal Tier Classification
- Correlation Analysis

TOTAL: 19 Algorithm Components
- 9 Core ML algorithms
- 3 Calibration methods
- 3 Supporting libraries
- 4 Quantum frameworks (optional)
"""

# Configuration
from .config import (
    MLConfig,
    SportConfig,
    BetType as ConfigBetType,
    default_ml_config,
    SPORT_CONFIGS,
)

# ELO Rating System
from .elo_rating import (
    ELOSystem,
    TeamELO,
    MultiSportELOManager,
)

# Feature Engineering
from .feature_engineering import (
    FeatureEngineer,
    FeatureSet,
    GameContext,
    BaseFeatureGenerator,
    FootballFeatureGenerator,
    BasketballFeatureGenerator,
    HockeyFeatureGenerator,
    BaseballFeatureGenerator,
    TennisFeatureGenerator,
)

# Training Frameworks
from .h2o_trainer import (
    H2OTrainer,
    H2OModelResult,
    H2OTrainerMock,
    get_h2o_trainer,
)

from .autogluon_trainer import (
    AutoGluonTrainer,
    AutoGluonModelResult,
    AutoGluonTrainerMock,
    get_autogluon_trainer,
)

from .sklearn_trainer import (
    SklearnEnsembleTrainer,
    SklearnModelResult,
    SklearnTrainerMock,
    get_sklearn_trainer,
)

# Deep Learning (TensorFlow/LSTM)
from .deep_learning_trainer import (
    DeepLearningTrainer,
    DeepLearningModelResult,
    DeepLearningTrainerMock,
    LSTMModel,
    HybridLSTMModel,
    DenseNeuralNetwork,
    get_deep_learning_trainer,
)

# Quantum ML (PennyLane, Qiskit, D-Wave)
from .quantum_ml import (
    QuantumMLTrainer,
    QuantumModelResult,
    QuantumMLTrainerMock,
    PennyLaneQNN,
    QiskitVQC,
    DWaveFeatureSelector,
    get_quantum_trainer,
    get_available_quantum_frameworks,
)

# Validation
from .walk_forward_validator import (
    WalkForwardValidator,
    WalkForwardResult,
    ValidationFold,
    ValidationMetrics,
    TimeSeriesSplitter,
)

# Probability Calibration
from .probability_calibration import (
    ProbabilityCalibrator,
    CalibrationResult,
    CalibrationMetrics,
)

# Meta-Ensemble
from .meta_ensemble import (
    MetaEnsemble,
    EnsembleWeights,
    EnsemblePrediction as MetaEnsemblePrediction,
    FrameworkPrediction as MetaFrameworkPrediction,
    WeightOptimizationResult,
    EnsemblePredictor,
)

# Model Registry
from .model_registry import (
    ModelRegistry,
    ModelVersion,
    ModelMetrics,
    TrainingConfig,
    PerformanceRecord,
    ModelStatus,
    ModelFramework as RegistryModelFramework,
)

# Advanced Prediction Engine
from .prediction_engine import (
    # Main Classes
    AdvancedPredictionEngine,
    PlayerPropsEngine,
    CorrelationAnalyzer,
    
    # Data Classes
    Prediction,
    PredictionBatch,
    PlayerPropPrediction,
    OddsInfo,
    FrameworkPrediction,
    EnsemblePrediction,
    SituationalModifiers,
    EdgeAnalysis,
    BettingRecommendation,
    PredictionExplanation,
    SportPredictionConfig,
    
    # Enums
    BetType,
    PredictedSide,
    SignalTier as PredictionSignalTier,
    MarketType,
    ModelFramework,
    
    # Constants
    SPORT_PREDICTION_CONFIGS,
    
    # Factory Functions
    create_advanced_prediction_engine,
    create_player_props_engine,
)

# SHAP Explanations
from .shap_explainer import (
    SHAPExplainer,
    SHAPExplainerMock,
    PredictionExplanation as SHAPPredictionExplanation,
    FeatureContribution,
    get_shap_explainer,
)

# Signal Tier Classification
from .signal_tier_classifier import (
    SignalTierClassifier,
    SignalTier,
    TierClassification,
    TierThresholds,
    TierPerformanceMetrics,
    assign_signal_tier,
)

# Ultimate Prediction System
from .ultimate_prediction_system import (
    UltimatePredictionSystem,
    SystemStatus,
    SystemHealth,
    ComponentHealth,
    PredictionRequest,
    PredictionResponse,
    DailyPredictionReport,
    create_ultimate_system,
    create_minimal_system,
)


__all__ = [
    # Configuration
    'MLConfig',
    'SportConfig',
    'ConfigBetType',
    'default_ml_config',
    'SPORT_CONFIGS',
    
    # ELO
    'ELOSystem',
    'TeamELO',
    'MultiSportELOManager',
    
    # Feature Engineering
    'FeatureEngineer',
    'FeatureSet',
    'GameContext',
    'BaseFeatureGenerator',
    'FootballFeatureGenerator',
    'BasketballFeatureGenerator',
    'HockeyFeatureGenerator',
    'BaseballFeatureGenerator',
    'TennisFeatureGenerator',
    
    # Trainers
    'H2OTrainer',
    'H2OModelResult',
    'H2OTrainerMock',
    'get_h2o_trainer',
    'AutoGluonTrainer',
    'AutoGluonModelResult',
    'AutoGluonTrainerMock',
    'get_autogluon_trainer',
    'SklearnEnsembleTrainer',
    'SklearnModelResult',
    'SklearnTrainerMock',
    'get_sklearn_trainer',
    
    # Deep Learning (TensorFlow/LSTM)
    'DeepLearningTrainer',
    'DeepLearningModelResult',
    'DeepLearningTrainerMock',
    'LSTMModel',
    'HybridLSTMModel',
    'DenseNeuralNetwork',
    'get_deep_learning_trainer',
    
    # Quantum ML
    'QuantumMLTrainer',
    'QuantumModelResult',
    'QuantumMLTrainerMock',
    'PennyLaneQNN',
    'QiskitVQC',
    'DWaveFeatureSelector',
    'get_quantum_trainer',
    'get_available_quantum_frameworks',
    
    # Validation
    'WalkForwardValidator',
    'WalkForwardResult',
    'ValidationFold',
    'ValidationMetrics',
    'TimeSeriesSplitter',
    
    # Calibration
    'ProbabilityCalibrator',
    'CalibrationResult',
    'CalibrationMetrics',
    
    # Meta-Ensemble
    'MetaEnsemble',
    'EnsembleWeights',
    'MetaEnsemblePrediction',
    'MetaFrameworkPrediction',
    'WeightOptimizationResult',
    'EnsemblePredictor',
    
    # Registry
    'ModelRegistry',
    'ModelVersion',
    'ModelMetrics',
    'TrainingConfig',
    'PerformanceRecord',
    'ModelStatus',
    'RegistryModelFramework',
    
    # Advanced Predictions
    'AdvancedPredictionEngine',
    'PlayerPropsEngine',
    'CorrelationAnalyzer',
    'Prediction',
    'PredictionBatch',
    'PlayerPropPrediction',
    'OddsInfo',
    'FrameworkPrediction',
    'EnsemblePrediction',
    'SituationalModifiers',
    'EdgeAnalysis',
    'BettingRecommendation',
    'PredictionExplanation',
    'SportPredictionConfig',
    'BetType',
    'PredictedSide',
    'PredictionSignalTier',
    'MarketType',
    'ModelFramework',
    'SPORT_PREDICTION_CONFIGS',
    'create_advanced_prediction_engine',
    'create_player_props_engine',
    
    # Explanations
    'SHAPExplainer',
    'SHAPExplainerMock',
    'SHAPPredictionExplanation',
    'FeatureContribution',
    'get_shap_explainer',
    
    # Signal Tiers
    'SignalTierClassifier',
    'SignalTier',
    'TierClassification',
    'TierThresholds',
    'TierPerformanceMetrics',
    'assign_signal_tier',
    
    # Ultimate System
    'UltimatePredictionSystem',
    'SystemStatus',
    'SystemHealth',
    'ComponentHealth',
    'PredictionRequest',
    'PredictionResponse',
    'DailyPredictionReport',
    'create_ultimate_system',
    'create_minimal_system',
    # Training Service
    'TrainingService',
    'TrainingResult',
    'get_training_service',
    'train_model_task',
]

# Training Service (lazy import to avoid circular dependencies)
try:
    from .training_service import (
        TrainingService,
        TrainingResult,
        get_training_service,
        train_model_task,
    )
except ImportError:
    # Training service may not be available in all contexts
    TrainingService = None
    TrainingResult = None
    get_training_service = None
    train_model_task = None

__version__ = '2.4.0'  # Added Training Service
