"""
LOYALEY - ML Configuration
Phase 2: ML Pipeline Configuration Settings
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class BetType(str, Enum):
    """Supported bet types"""
    SPREAD = "spread"
    MONEYLINE = "moneyline"
    TOTAL = "total"
    PLAYER_PROP = "player_prop"


class Framework(str, Enum):
    """ML frameworks"""
    H2O = "h2o"
    AUTOGLUON = "autogluon"
    SKLEARN = "sklearn"


class SignalTier(str, Enum):
    """Prediction signal tiers"""
    A = "A"  # >= 65% confidence
    B = "B"  # 60-65% confidence
    C = "C"  # 55-60% confidence
    D = "D"  # < 55% confidence


@dataclass
class SportConfig:
    """Sport-specific ML configuration"""
    code: str
    name: str
    api_key: str
    feature_count: int
    elo_k_factor: int
    elo_home_advantage: int
    bet_types: List[BetType]
    has_player_props: bool = True
    rolling_windows: List[int] = field(default_factory=lambda: [3, 5, 10, 15, 30])
    

# Sport configurations with ELO parameters from documentation
SPORT_CONFIGS: Dict[str, SportConfig] = {
    "NFL": SportConfig(
        code="NFL",
        name="NFL Football",
        api_key="americanfootball_nfl",
        feature_count=75,
        elo_k_factor=20,
        elo_home_advantage=48,
        bet_types=[BetType.SPREAD, BetType.MONEYLINE, BetType.TOTAL],
        has_player_props=True,
    ),
    "NCAAF": SportConfig(
        code="NCAAF",
        name="NCAA Football",
        api_key="americanfootball_ncaaf",
        feature_count=70,
        elo_k_factor=32,
        elo_home_advantage=70,
        bet_types=[BetType.SPREAD, BetType.MONEYLINE, BetType.TOTAL],
        has_player_props=True,
    ),
    "CFL": SportConfig(
        code="CFL",
        name="CFL Football",
        api_key="americanfootball_cfl",
        feature_count=65,
        elo_k_factor=20,
        elo_home_advantage=48,
        bet_types=[BetType.SPREAD, BetType.MONEYLINE, BetType.TOTAL],
        has_player_props=False,
    ),
    "NBA": SportConfig(
        code="NBA",
        name="NBA Basketball",
        api_key="basketball_nba",
        feature_count=80,
        elo_k_factor=20,
        elo_home_advantage=100,
        bet_types=[BetType.SPREAD, BetType.MONEYLINE, BetType.TOTAL],
        has_player_props=True,
    ),
    "NCAAB": SportConfig(
        code="NCAAB",
        name="NCAA Basketball",
        api_key="basketball_ncaab",
        feature_count=70,
        elo_k_factor=32,
        elo_home_advantage=100,
        bet_types=[BetType.SPREAD, BetType.MONEYLINE, BetType.TOTAL],
        has_player_props=True,
    ),
    "WNBA": SportConfig(
        code="WNBA",
        name="WNBA Basketball",
        api_key="basketball_wnba",
        feature_count=70,
        elo_k_factor=20,
        elo_home_advantage=80,
        bet_types=[BetType.SPREAD, BetType.MONEYLINE, BetType.TOTAL],
        has_player_props=True,
    ),
    "NHL": SportConfig(
        code="NHL",
        name="NHL Hockey",
        api_key="icehockey_nhl",
        feature_count=75,
        elo_k_factor=8,
        elo_home_advantage=33,
        bet_types=[BetType.SPREAD, BetType.MONEYLINE, BetType.TOTAL],
        has_player_props=True,
    ),
    "MLB": SportConfig(
        code="MLB",
        name="MLB Baseball",
        api_key="baseball_mlb",
        feature_count=85,
        elo_k_factor=4,
        elo_home_advantage=24,
        bet_types=[BetType.SPREAD, BetType.MONEYLINE, BetType.TOTAL],
        has_player_props=True,
    ),
    "ATP": SportConfig(
        code="ATP",
        name="ATP Tennis",
        api_key="tennis_atp",
        feature_count=60,
        elo_k_factor=32,
        elo_home_advantage=0,  # No home advantage in tennis
        bet_types=[BetType.MONEYLINE],
        has_player_props=False,
    ),
    "WTA": SportConfig(
        code="WTA",
        name="WTA Tennis",
        api_key="tennis_wta",
        feature_count=60,
        elo_k_factor=32,
        elo_home_advantage=0,  # No home advantage in tennis
        bet_types=[BetType.MONEYLINE],
        has_player_props=False,
    ),
}


@dataclass
class MLConfig:
    """Main ML configuration class"""
    
    # H2O AutoML settings
    h2o_max_models: int = 50
    h2o_max_runtime_secs: int = 3600
    h2o_max_mem_size: str = "32g"
    h2o_nfolds: int = 5
    h2o_seed: int = 42
    
    # AutoGluon settings
    autogluon_presets: str = "best_quality"
    autogluon_time_limit: int = 3600
    autogluon_num_bag_folds: int = 8
    autogluon_num_stack_levels: int = 2
    autogluon_eval_metric: str = "roc_auc"
    
    # Sklearn ensemble settings
    sklearn_n_estimators: int = 500
    sklearn_max_depth: int = 8
    sklearn_learning_rate: float = 0.05
    sklearn_subsample: float = 0.8
    sklearn_colsample_bytree: float = 0.8
    
    # Walk-forward validation settings
    training_window_days: int = 365
    validation_window_days: int = 30
    step_size_days: int = 30
    min_training_size_days: int = 180
    gap_days: int = 1  # Gap between training and validation
    
    # Probability calibration settings
    calibration_method: str = "isotonic"  # isotonic, platt, temperature
    calibration_cv_folds: int = 5
    
    # Signal tier thresholds
    tier_a_threshold: float = 0.65
    tier_b_threshold: float = 0.60
    tier_c_threshold: float = 0.55
    
    # Meta-ensemble settings
    ensemble_min_weight: float = 0.1
    ensemble_weight_decay: float = 0.95
    
    # Model management
    model_artifact_path: str = "./models"
    max_models_per_sport: int = 5
    model_ttl_days: int = 90
    
    # SHAP settings
    shap_max_samples: int = 1000
    shap_top_features: int = 10
    
    # Feature engineering
    default_rolling_windows: List[int] = field(
        default_factory=lambda: [3, 5, 10, 15, 30]
    )
    elo_base_rating: float = 1500.0
    elo_scale: float = 400.0
    momentum_decay_factor: float = 0.9
    
    # Performance targets
    target_accuracy_overall: float = 0.60
    target_accuracy_tier_a: float = 0.65
    target_auc: float = 0.60
    target_log_loss: float = 0.68
    target_brier_score: float = 0.24
    target_calibration_error: float = 0.05
    
    # Training schedule
    weekly_retrain_day: int = 0  # Monday
    weekly_retrain_hour: int = 4  # 4 AM UTC
    
    def get_sport_config(self, sport_code: str) -> SportConfig:
        """Get configuration for a specific sport"""
        if sport_code not in SPORT_CONFIGS:
            raise ValueError(f"Unknown sport code: {sport_code}")
        return SPORT_CONFIGS[sport_code]
    
    def get_all_sports(self) -> List[str]:
        """Get all supported sport codes"""
        return list(SPORT_CONFIGS.keys())
    
    @classmethod
    def from_env(cls) -> 'MLConfig':
        """Create config from environment variables"""
        import os
        return cls(
            h2o_max_models=int(os.getenv('H2O_MAX_MODELS', 50)),
            h2o_max_runtime_secs=int(os.getenv('H2O_MAX_RUNTIME_SECS', 3600)),
            h2o_max_mem_size=os.getenv('H2O_MAX_MEM_SIZE', '32g'),
            autogluon_time_limit=int(os.getenv('AUTOGLUON_TIME_LIMIT', 3600)),
            tier_a_threshold=float(os.getenv('SIGNAL_TIER_A_MIN', 0.65)),
            tier_b_threshold=float(os.getenv('SIGNAL_TIER_B_MIN', 0.60)),
            tier_c_threshold=float(os.getenv('SIGNAL_TIER_C_MIN', 0.55)),
        )


# Default configuration instance
default_ml_config = MLConfig()
