"""
ROYALEY - Phase 4 Enterprise Configuration
Complete configuration management with validation and secrets handling
"""

import os
import secrets
from typing import Any, Dict, List, Optional
from functools import lru_cache
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Enterprise-grade configuration settings"""
    
    # Application Settings
    APP_NAME: str = "Royaley"
    APP_VERSION: str = "2.0.0"
    ENVIRONMENT: str = Field(default="development", pattern="^(development|staging|production)$")
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    
    # Property aliases for lowercase access
    @property
    def app_name(self) -> str:
        return self.APP_NAME
    
    @property
    def app_version(self) -> str:
        return self.APP_VERSION
    
    @property
    def debug(self) -> bool:
        return self.DEBUG
    
    @property
    def port(self) -> int:
        return self.PORT
    
    @property
    def workers(self) -> int:
        return self.WORKERS
    
    @property
    def cors_origins(self) -> List[str]:
        return self.CORS_ORIGINS
    
    @property
    def rate_limit_requests(self) -> int:
        return self.RATE_LIMIT_REQUESTS
    
    @property
    def environment(self) -> str:
        return self.ENVIRONMENT
    
    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 4
    RELOAD: bool = False
    
    # Security Settings
    SECRET_KEY: str = Field(default_factory=lambda: secrets.token_urlsafe(64))
    JWT_SECRET_KEY: str = Field(default_factory=lambda: secrets.token_urlsafe(64))
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    BCRYPT_ROUNDS: int = 12
    
    # Encryption Settings
    AES_KEY: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    ENCRYPTION_ENABLED: bool = True
    
    # Two-Factor Authentication
    TOTP_ENABLED: bool = True
    TOTP_ISSUER: str = "Royaley"
    TOTP_TIME_STEP: int = 30
    
    # Database Settings
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/royaley"
    DATABASE_POOL_SIZE: int = 20
    DATABASE_MAX_OVERFLOW: int = 10
    DATABASE_POOL_TIMEOUT: int = 30
    DATABASE_POOL_RECYCLE: int = 3600
    DATABASE_ECHO: bool = False
    
    # Redis Settings
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_MAX_CONNECTIONS: int = 50
    REDIS_SOCKET_TIMEOUT: float = 5.0
    CACHE_TTL_DEFAULT: int = 300
    CACHE_TTL_PREDICTIONS: int = 60
    CACHE_TTL_ODDS: int = 30
    CACHE_TTL_MODELS: int = 3600
    
    # External API Settings
    ODDS_API_KEY: str = "8508537db871e84356d777894318f6c3"
    ODDS_API_BASE_URL: str = "https://api.the-odds-api.com/v4"
    ODDS_API_RATE_LIMIT: int = 500
    ESPN_API_BASE_URL: str = "https://site.api.espn.com/apis/site/v2"
    
    # RapidAPI (Pinnacle Odds, etc.)
    RAPIDAPI_KEY: str = ""
    PINNACLE_API_HOST: str = "pinnacle-odds-api.p.rapidapi.com"
    
    # Weather API
    WEATHER_API_KEY: str = ""
    WEATHER_API_BASE_URL: str = "https://api.openweathermap.org/data/2.5"
    
    # TheSportsDB API
    SPORTSDB_API_KEY: str = "688655"
    SPORTSDB_BASE_URL: str = "https://www.thesportsdb.com/api/v2/json"
    SPORTSDB_V1_BASE_URL: str = "https://www.thesportsdb.com/api/v1/json"
    
    # Google Maps API (for travel/distance)
    GOOGLE_MAPS_API_KEY: str = ""
    
    # BallDontLie API
    BALLDONTLIE_API_KEY: str = ""
    BALLDONTLIE_BASE_URL: str = "https://api.balldontlie.io/v1"
    
    # ML Configuration
    H2O_MAX_MEM_SIZE: str = "32g"
    H2O_MAX_MODELS: int = 50
    H2O_MAX_RUNTIME_SECS: int = 3600
    H2O_SEED: int = 42
    AUTOGLUON_TIME_LIMIT: int = 3600
    AUTOGLUON_PRESETS: str = "best_quality"
    MODEL_STORAGE_PATH: str = "models/"
    
    # Betting Configuration
    KELLY_FRACTION: float = 0.25
    MAX_BET_PERCENT: float = 0.02
    MIN_EDGE_THRESHOLD: float = 0.03
    MIN_BET_SIZE: float = 10.0
    DEFAULT_BANKROLL: float = 10000.0
    
    # Signal Tier Thresholds
    SIGNAL_TIER_A_MIN: float = 0.65
    SIGNAL_TIER_B_MIN: float = 0.60
    SIGNAL_TIER_C_MIN: float = 0.55
    
    # Walk-Forward Validation
    WFV_TRAINING_WINDOW_DAYS: int = 365
    WFV_TEST_WINDOW_DAYS: int = 30
    WFV_STEP_SIZE_DAYS: int = 30
    WFV_MIN_TRAINING_DAYS: int = 180
    
    # Monitoring Settings
    PROMETHEUS_PORT: int = 9090
    GRAFANA_PORT: int = 3000
    METRICS_ENABLED: bool = True
    HEALTH_CHECK_INTERVAL: int = 60
    
    # Alerting Settings
    TELEGRAM_BOT_TOKEN: str = ""
    TELEGRAM_CHAT_ID: str = ""
    SLACK_WEBHOOK_URL: str = ""
    EMAIL_SMTP_HOST: str = ""
    EMAIL_SMTP_PORT: int = 587
    EMAIL_SMTP_USER: str = ""
    EMAIL_SMTP_PASSWORD: str = ""
    EMAIL_FROM_ADDRESS: str = ""
    ALERT_EMAIL_RECIPIENTS: List[str] = []
    DATADOG_API_KEY: str = ""
    PAGERDUTY_API_KEY: str = ""
    
    # Self-Healing Settings
    SELF_HEALING_ENABLED: bool = True
    AUTO_RESTART_FAILED_SERVICES: bool = True
    MAX_RESTART_ATTEMPTS: int = 3
    RESTART_COOLDOWN_SECONDS: int = 60
    CIRCUIT_BREAKER_THRESHOLD: int = 5
    CIRCUIT_BREAKER_TIMEOUT: int = 30
    
    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 60
    
    # CORS Settings
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8080"]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: List[str] = ["*"]
    CORS_ALLOW_HEADERS: List[str] = ["*"]
    
    # Supported Sports
    SUPPORTED_SPORTS: List[str] = ["NFL", "NCAAF", "CFL", "NBA", "NCAAB", "WNBA", "NHL", "MLB", "ATP", "WTA"]
    
    # Scheduler Settings
    SCHEDULER_ENABLED: bool = True
    ODDS_REFRESH_INTERVAL: int = 60
    GAMES_REFRESH_INTERVAL: int = 300
    GRADING_INTERVAL: int = 900
    PREDICTION_GENERATION_INTERVAL: int = 3600
    MODEL_RETRAINING_CRON: str = "0 4 * * *"
    DATA_CLEANUP_CRON: str = "0 2 * * *"
    
    # Backup Settings
    BACKUP_ENABLED: bool = True
    BACKUP_RETENTION_DAYS: int = 30
    BACKUP_PATH: str = "/backups"
    S3_BACKUP_BUCKET: str = ""
    S3_ACCESS_KEY: str = ""
    S3_SECRET_KEY: str = ""
    
    # Feature Flags
    FEATURE_PLAYER_PROPS: bool = True
    FEATURE_LIVE_BETTING: bool = False
    FEATURE_SHAP_EXPLANATIONS: bool = True
    FEATURE_ADVANCED_ANALYTICS: bool = True
    
    @field_validator('DATABASE_URL')
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        if not v.startswith(('postgresql://', 'postgresql+asyncpg://')):
            raise ValueError('DATABASE_URL must be a PostgreSQL connection string')
        return v
    
    @field_validator('REDIS_URL')
    @classmethod
    def validate_redis_url(cls, v: str) -> str:
        if not v.startswith('redis://'):
            raise ValueError('REDIS_URL must be a valid Redis connection string')
        return v
    
    @field_validator('KELLY_FRACTION')
    @classmethod
    def validate_kelly_fraction(cls, v: float) -> float:
        if not 0 < v <= 1:
            raise ValueError('KELLY_FRACTION must be between 0 and 1')
        return v
    
    @model_validator(mode='after')
    def validate_production_settings(self) -> 'Settings':
        """Ensure production has proper security settings"""
        if self.ENVIRONMENT == 'production':
            if self.DEBUG:
                raise ValueError('DEBUG must be False in production')
            if len(self.SECRET_KEY) < 32:
                raise ValueError('SECRET_KEY must be at least 32 characters in production')
        return self
    
    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"
    
    @property
    def is_development(self) -> bool:
        return self.ENVIRONMENT == "development"
    
    def get_database_url(self, async_driver: bool = True) -> str:
        """Get database URL with appropriate driver"""
        if async_driver and 'asyncpg' not in self.DATABASE_URL:
            return self.DATABASE_URL.replace('postgresql://', 'postgresql+asyncpg://')
        return self.DATABASE_URL
    
    def get_sports_config(self, sport_code: str) -> Dict[str, Any]:
        """Get sport-specific configuration"""
        configs = {
            "NFL": {"features": 75, "k_factor": 32, "home_advantage": 2.5},
            "NCAAF": {"features": 70, "k_factor": 28, "home_advantage": 3.0},
            "CFL": {"features": 65, "k_factor": 28, "home_advantage": 3.0},
            "NBA": {"features": 80, "k_factor": 20, "home_advantage": 3.0},
            "NCAAB": {"features": 70, "k_factor": 24, "home_advantage": 3.5},
            "WNBA": {"features": 70, "k_factor": 24, "home_advantage": 2.5},
            "NHL": {"features": 75, "k_factor": 16, "home_advantage": 0.08},
            "MLB": {"features": 85, "k_factor": 12, "home_advantage": 0.04},
            "ATP": {"features": 60, "k_factor": 32, "home_advantage": 0},
            "WTA": {"features": 60, "k_factor": 32, "home_advantage": 0},
        }
        return configs.get(sport_code.upper(), {"features": 60, "k_factor": 20, "home_advantage": 2.0})
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Singleton instance
settings = get_settings()


# Sport code mapping for TheOddsAPI
ODDS_API_SPORT_KEYS = {
    "NFL": "americanfootball_nfl",
    "NCAAF": "americanfootball_ncaaf",
    "CFL": "americanfootball_cfl",
    "NBA": "basketball_nba",
    "NCAAB": "basketball_ncaab",
    "WNBA": "basketball_wnba",
    "NHL": "icehockey_nhl",
    "MLB": "baseball_mlb",
    "ATP": "tennis_atp_singles",
    "WTA": "tennis_wta_singles",
}
