"""
LOYALEY - Services Module
Complete service layer for the enterprise sports prediction platform.
"""

# ML Services
from app.services.ml.feature_engineering import FeatureEngineer
from app.services.ml.h2o_trainer import H2OTrainer
from app.services.ml.autogluon_trainer import AutoGluonTrainer
from app.services.ml.sklearn_trainer import SklearnEnsembleTrainer
from app.services.ml.meta_ensemble import MetaEnsemble
from app.services.ml.prediction_engine import PredictionEngine
from app.services.ml.probability_calibration import ProbabilityCalibrator
from app.services.ml.signal_tier_classifier import SignalTierClassifier
from app.services.ml.walk_forward_validator import WalkForwardValidator
from app.services.ml.shap_explainer import SHAPExplainer
from app.services.ml.elo_rating import ELORatingSystem
from app.services.ml.model_registry import ModelRegistry

# Betting Services
from app.services.betting.kelly_calculator import KellyCalculator
from app.services.betting.clv_calculator import CLVCalculator
from app.services.betting.auto_grader import AutoGrader
from app.services.betting.line_movement_analyzer import LineMovementAnalyzer

# Data Collection Services
from app.services.collectors.odds_collector import OddsCollector
from app.services.collectors.espn_collector import ESPNCollector
from app.services.collectors.tennis_collector import TennisCollector

# Enterprise Services
from app.services.self_healing.self_healing_service import SelfHealingService, self_healing_service
from app.services.alerting.alerting_service import AlertingService, alerting_service
from app.services.monitoring.metrics_service import MonitoringService, monitoring_service
from app.services.scheduling.scheduler_service import SchedulerService, scheduler_service
from app.services.data_quality.data_quality_service import DataQualityService, data_quality_service

# Backtesting Services
from app.services.backtesting.backtest_engine import BacktestEngine
from app.services.backtesting.walk_forward import WalkForwardEngine
from app.services.backtesting.simulation import BettingSimulator

# Integrity Services
from app.services.integrity.sha256_lockln import PredictionIntegrity
from app.services.integrity.shap_explainer import PredictionExplainer

__all__ = [
    # ML Services
    "FeatureEngineer",
    "H2OTrainer",
    "AutoGluonTrainer",
    "SklearnEnsembleTrainer",
    "MetaEnsemble",
    "PredictionEngine",
    "ProbabilityCalibrator",
    "SignalTierClassifier",
    "WalkForwardValidator",
    "SHAPExplainer",
    "ELORatingSystem",
    "ModelRegistry",
    
    # Betting Services
    "KellyCalculator",
    "CLVCalculator",
    "AutoGrader",
    "LineMovementAnalyzer",
    
    # Data Collection
    "OddsCollector",
    "ESPNCollector",
    "TennisCollector",
    
    # Enterprise Services
    "SelfHealingService",
    "self_healing_service",
    "AlertingService",
    "alerting_service",
    "MonitoringService",
    "monitoring_service",
    "SchedulerService",
    "scheduler_service",
    "DataQualityService",
    "data_quality_service",
    
    # Backtesting
    "BacktestEngine",
    "WalkForwardEngine",
    "BettingSimulator",
    
    # Integrity
    "PredictionIntegrity",
    "PredictionExplainer",
]

# Web Scrapers
from .scrapers import (
    get_scraper,
    SCRAPER_REGISTRY,
    OddsScraper,
    ScoresScraper,
    TeamStatsScraper,
    PlayerStatsScraper,
    InjuriesScraper,
    SchedulesScraper,
    StandingsScraper,
    PlayerPropsScraper,
    LineMovementsScraper,
    GenericScraper,
)
