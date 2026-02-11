"""
ROYALEY - Database Models
Complete SQLAlchemy 2.0 models for the enterprise sports prediction platform.
57 tables: 45 original + 12 master data unification tables.
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

# Import injury models (44-45)
from app.models.injury_models import (
    Injury,
    GameInjury,
)

# Import master data models (46-57)
from app.models.master_data_models import (
    SourceRegistry,
    MappingAuditLog,
    MasterTeam,
    MasterPlayer,
    MasterGame,
    TeamMapping,
    PlayerMapping,
    GameMapping,
    VenueMapping,
    MasterOdds,
    OddsMapping,
    MLTrainingDataset,
)

# Import upcoming/live pipeline models (58-59)
from app.models.upcoming_models import (
    UpcomingGame,
    UpcomingOdds,
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

    # Injury Tables (44-45)
    "Injury",
    "GameInjury",

    # Master Data Architecture (46-57)
    "SourceRegistry",
    "MappingAuditLog",
    "MasterTeam",
    "MasterPlayer",
    "MasterGame",
    "TeamMapping",
    "PlayerMapping",
    "GameMapping",
    "VenueMapping",
    "MasterOdds",
    "OddsMapping",
    "MLTrainingDataset",

    # Live Pipeline Tables (58-59)
    "UpcomingGame",
    "UpcomingOdds",
]