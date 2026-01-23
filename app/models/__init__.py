"""
ROYALEY - Database Models
Complete SQLAlchemy 2.0 models for the enterprise sports prediction platform.
43 tables supporting all system functionality.
"""

from app.models.models import (
    # Base
    Base,
    
    # Enums
    UserRole,
    GameStatus,
    BetResult,
    SignalTier,
    AlertSeverity,
    HealthStatus,
    MLFramework,
    TaskStatus,
    
    # Users & Authentication
    User,
    Session,
    APIKey,
    UserPreference,
    AuditLog,
    
    # Sports Data
    Sport,
    Team,
    Player,
    Venue,
    Season,
    Game,
    GameFeature,
    TeamStats,
    PlayerStats,
    
    # Odds & Markets
    Sportsbook,
    Odds,
    OddsMovement,
    ClosingLine,
    ConsensusLine,
    
    # Predictions
    Prediction,
    PredictionResult,
    PlayerProp,
    ShapExplanation,
    
    # ML Models
    MLModel,
    TrainingRun,
    ModelPerformance,
    FeatureImportance,
    CalibrationModel,
    
    # Betting
    Bankroll,
    Bet,
    BankrollTransaction,
    
    # System
    SystemSetting,
    ScheduledTask,
    Alert,
    DataQualityCheck,
    SystemHealthSnapshot,
    BacktestRun,
    
    # Additional Tracking
    ELOHistory,
    CLVRecord,
    LineMovementAlert,
    
    # Additional Tables (41-43)
    Notification,
    RateLimitLog,
    WeatherData,
)

__all__ = [
    # Base
    "Base",
    
    # Enums
    "UserRole",
    "GameStatus",
    "BetResult",
    "SignalTier",
    "AlertSeverity",
    "HealthStatus",
    "MLFramework",
    "TaskStatus",
    
    # Users & Authentication
    "User",
    "Session",
    "APIKey",
    "UserPreference",
    "AuditLog",
    
    # Sports Data
    "Sport",
    "Team",
    "Player",
    "Venue",
    "Season",
    "Game",
    "GameFeature",
    "TeamStats",
    "PlayerStats",
    
    # Odds & Markets
    "Sportsbook",
    "Odds",
    "OddsMovement",
    "ClosingLine",
    "ConsensusLine",
    
    # Predictions
    "Prediction",
    "PredictionResult",
    "PlayerProp",
    "ShapExplanation",
    
    # ML Models
    "MLModel",
    "TrainingRun",
    "ModelPerformance",
    "FeatureImportance",
    "CalibrationModel",
    
    # Betting
    "Bankroll",
    "Bet",
    "BankrollTransaction",
    
    # System
    "SystemSetting",
    "ScheduledTask",
    "Alert",
    "DataQualityCheck",
    "SystemHealthSnapshot",
    "BacktestRun",
    
    # Additional Tracking
    "ELOHistory",
    "CLVRecord",
    "LineMovementAlert",
    
    # Additional Tables
    "Notification",
    "RateLimitLog",
    "WeatherData",
]
