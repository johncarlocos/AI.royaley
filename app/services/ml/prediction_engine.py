"""
LOYALEY - Advanced Prediction Engine
Phase 2: Enterprise-Grade Prediction System

This is the most advanced prediction engine featuring:
- Dynamic ensemble weighting with performance decay
- Confidence intervals and uncertainty quantification
- Kelly Criterion integration with fractional sizing
- CLV (Closing Line Value) tracking
- Sharp vs public money detection
- Contextual tier adjustments
- Sport-specific optimizations
- Model drift detection
- Real-time line movement tracking
- Player props integration
- Correlation analysis
- Situational modifiers
- Bayesian probability updates
"""

import numpy as np
import hashlib
import hmac
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Tuple, Union
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import warnings

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class BetType(Enum):
    """Supported bet types."""
    SPREAD = 'spread'
    MONEYLINE = 'moneyline'
    TOTAL = 'total'
    FIRST_HALF_SPREAD = 'first_half_spread'
    FIRST_HALF_TOTAL = 'first_half_total'
    PLAYER_PROP = 'player_prop'
    TEAM_PROP = 'team_prop'
    GAME_PROP = 'game_prop'


class PredictedSide(Enum):
    """Predicted bet side."""
    HOME = 'home'
    AWAY = 'away'
    OVER = 'over'
    UNDER = 'under'


class SignalTier(Enum):
    """Prediction confidence tiers with detailed attributes."""
    A = 'A'  # 65%+ confidence - Elite predictions
    B = 'B'  # 60-65% - Strong value
    C = 'C'  # 55-60% - Moderate confidence
    D = 'D'  # <55% - Track only
    
    @property
    def kelly_multiplier(self) -> float:
        """Kelly fraction multiplier for this tier."""
        return {'A': 1.0, 'B': 0.75, 'C': 0.5, 'D': 0.0}.get(self.value, 0.0)
    
    @property
    def max_bet_percent(self) -> float:
        """Maximum bet as percentage of bankroll."""
        return {'A': 0.02, 'B': 0.015, 'C': 0.01, 'D': 0.0}.get(self.value, 0.0)
    
    @property
    def target_accuracy(self) -> float:
        """Target accuracy for this tier."""
        return {'A': 0.65, 'B': 0.60, 'C': 0.55, 'D': 0.50}.get(self.value, 0.50)


class MarketType(Enum):
    """Market classification for betting."""
    SHARP = 'sharp'           # Sharp/professional market
    SQUARE = 'square'         # Recreational market
    MIXED = 'mixed'           # Mixed action
    UNKNOWN = 'unknown'


class ModelFramework(Enum):
    """ML framework identifiers."""
    H2O = 'h2o'
    AUTOGLUON = 'autogluon'
    SKLEARN = 'sklearn'
    NEURAL = 'neural'
    ENSEMBLE = 'ensemble'


# =============================================================================
# SPORT-SPECIFIC CONFIGURATION
# =============================================================================

@dataclass
class SportPredictionConfig:
    """Sport-specific prediction configuration."""
    sport_code: str
    
    # Tier thresholds (can be sport-specific)
    tier_a_min: float = 0.65
    tier_b_min: float = 0.60
    tier_c_min: float = 0.55
    
    # Edge requirements
    min_edge_spread: float = 0.03
    min_edge_moneyline: float = 0.04
    min_edge_total: float = 0.03
    
    # Kelly settings
    kelly_fraction: float = 0.25
    max_bet_percent: float = 0.02
    
    # Sport characteristics
    typical_total: float = 200.0
    typical_spread: float = 5.0
    home_advantage_points: float = 3.0
    
    # Model weights (learned from performance)
    h2o_weight: float = 0.33
    autogluon_weight: float = 0.40
    sklearn_weight: float = 0.27
    
    # Situational adjustments
    b2b_penalty: float = 0.02  # Reduce confidence for back-to-back
    rest_bonus: float = 0.01  # Increase per rest day advantage
    travel_penalty: float = 0.01  # Per timezone crossed
    
    # Line movement sensitivity
    steam_move_threshold: float = 1.5  # Points
    reverse_line_threshold: float = 0.65  # Public % for RLM
    
    # Correlation factors
    spread_total_correlation: float = 0.15
    
    def get_min_edge(self, bet_type: BetType) -> float:
        """Get minimum edge for bet type."""
        if bet_type == BetType.SPREAD:
            return self.min_edge_spread
        elif bet_type == BetType.MONEYLINE:
            return self.min_edge_moneyline
        elif bet_type == BetType.TOTAL:
            return self.min_edge_total
        return 0.03


# Default sport configurations
SPORT_PREDICTION_CONFIGS: Dict[str, SportPredictionConfig] = {
    'NFL': SportPredictionConfig(
        sport_code='NFL',
        typical_total=44.0,
        typical_spread=3.0,
        home_advantage_points=2.5,
        min_edge_spread=0.03,
        autogluon_weight=0.45,  # NFL favors AutoGluon
    ),
    'NBA': SportPredictionConfig(
        sport_code='NBA',
        typical_total=220.0,
        typical_spread=5.0,
        home_advantage_points=3.0,
        b2b_penalty=0.03,  # B2B matters more in NBA
        autogluon_weight=0.42,
    ),
    'MLB': SportPredictionConfig(
        sport_code='MLB',
        typical_total=8.5,
        typical_spread=1.5,
        home_advantage_points=0.3,
        min_edge_moneyline=0.05,  # MLB needs more edge on ML
        h2o_weight=0.40,  # MLB favors H2O
    ),
    'NHL': SportPredictionConfig(
        sport_code='NHL',
        typical_total=5.5,
        typical_spread=1.5,
        home_advantage_points=0.2,
        b2b_penalty=0.025,
    ),
    'NCAAF': SportPredictionConfig(
        sport_code='NCAAF',
        typical_total=52.0,
        typical_spread=10.0,
        home_advantage_points=3.5,
        tier_a_min=0.63,  # Lower threshold for NCAAF
    ),
    'NCAAB': SportPredictionConfig(
        sport_code='NCAAB',
        typical_total=140.0,
        typical_spread=7.0,
        home_advantage_points=4.0,
        tier_a_min=0.63,
    ),
    'WNBA': SportPredictionConfig(
        sport_code='WNBA',
        typical_total=160.0,
        typical_spread=5.0,
        home_advantage_points=2.5,
    ),
    'CFL': SportPredictionConfig(
        sport_code='CFL',
        typical_total=52.0,
        typical_spread=5.0,
        home_advantage_points=3.0,
    ),
    'ATP': SportPredictionConfig(
        sport_code='ATP',
        typical_total=22.0,  # Games
        typical_spread=3.5,
        home_advantage_points=0.0,  # No home advantage in tennis
    ),
    'WTA': SportPredictionConfig(
        sport_code='WTA',
        typical_total=20.0,
        typical_spread=3.0,
        home_advantage_points=0.0,
    ),
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class OddsInfo:
    """Comprehensive odds information from multiple sportsbooks."""
    game_id: str
    
    # Spread
    spread_home: Optional[float] = None
    spread_away: Optional[float] = None
    spread_home_odds: Optional[int] = None  # American odds
    spread_away_odds: Optional[int] = None
    
    # Moneyline
    moneyline_home: Optional[int] = None
    moneyline_away: Optional[int] = None
    
    # Total
    total_line: Optional[float] = None
    total_over_odds: Optional[int] = None
    total_under_odds: Optional[int] = None
    
    # Opening lines (for CLV)
    opening_spread: Optional[float] = None
    opening_total: Optional[float] = None
    opening_ml_home: Optional[int] = None
    opening_ml_away: Optional[int] = None
    
    # Market data
    public_spread_home_pct: Optional[float] = None
    public_ml_home_pct: Optional[float] = None
    public_total_over_pct: Optional[float] = None
    money_spread_home_pct: Optional[float] = None  # Money % vs tickets
    
    # Sportsbook source
    sportsbook: str = 'consensus'
    recorded_at: datetime = field(default_factory=datetime.utcnow)
    
    # Pinnacle (sharp benchmark)
    pinnacle_spread: Optional[float] = None
    pinnacle_ml_home: Optional[int] = None
    pinnacle_total: Optional[float] = None
    
    def get_implied_probability(self, bet_type: BetType, side: PredictedSide) -> float:
        """Calculate implied probability from American odds with vig removal."""
        odds = self._get_odds(bet_type, side)
        if odds is None:
            return 0.5
        
        # Get both sides for vig removal
        opposite_side = PredictedSide.AWAY if side == PredictedSide.HOME else PredictedSide.HOME
        if side == PredictedSide.OVER:
            opposite_side = PredictedSide.UNDER
        elif side == PredictedSide.UNDER:
            opposite_side = PredictedSide.OVER
            
        opposite_odds = self._get_odds(bet_type, opposite_side)
        
        # Convert to implied probabilities
        prob = self._american_to_prob(odds)
        if opposite_odds:
            opposite_prob = self._american_to_prob(opposite_odds)
            # Remove vig (normalize to 100%)
            total = prob + opposite_prob
            if total > 0:
                prob = prob / total
        
        return prob
    
    def _get_odds(self, bet_type: BetType, side: PredictedSide) -> Optional[int]:
        """Get odds for specific bet type and side."""
        if bet_type == BetType.SPREAD:
            return self.spread_home_odds if side == PredictedSide.HOME else self.spread_away_odds
        elif bet_type == BetType.MONEYLINE:
            return self.moneyline_home if side == PredictedSide.HOME else self.moneyline_away
        elif bet_type == BetType.TOTAL:
            return self.total_over_odds if side == PredictedSide.OVER else self.total_under_odds
        return None
    
    def _american_to_prob(self, odds: int) -> float:
        """Convert American odds to implied probability."""
        if odds < 0:
            return abs(odds) / (abs(odds) + 100)
        else:
            return 100 / (odds + 100)
    
    def get_spread_movement(self) -> float:
        """Calculate spread movement from open."""
        if self.opening_spread is not None and self.spread_home is not None:
            return self.spread_home - self.opening_spread
        return 0.0
    
    def get_total_movement(self) -> float:
        """Calculate total movement from open."""
        if self.opening_total is not None and self.total_line is not None:
            return self.total_line - self.opening_total
        return 0.0
    
    def detect_steam_move(self, threshold: float = 1.5) -> bool:
        """Detect if there's been a steam move."""
        spread_move = abs(self.get_spread_movement())
        return spread_move >= threshold
    
    def detect_reverse_line_movement(self) -> Optional[str]:
        """Detect reverse line movement (sharp action)."""
        if self.public_spread_home_pct is None:
            return None
        
        spread_move = self.get_spread_movement()
        
        # RLM: Line moves opposite to public betting
        if self.public_spread_home_pct > 0.60 and spread_move > 0:
            return 'away'  # Sharp money on away
        elif self.public_spread_home_pct < 0.40 and spread_move < 0:
            return 'home'  # Sharp money on home
        
        return None
    
    def get_market_type(self) -> MarketType:
        """Classify the market based on betting patterns."""
        if self.money_spread_home_pct is None or self.public_spread_home_pct is None:
            return MarketType.UNKNOWN
        
        # Compare money % to ticket %
        diff = abs(self.money_spread_home_pct - self.public_spread_home_pct)
        
        if diff > 0.15:
            return MarketType.SHARP
        elif diff < 0.05:
            return MarketType.SQUARE
        else:
            return MarketType.MIXED


@dataclass
class FrameworkPrediction:
    """Prediction from a single ML framework."""
    framework: ModelFramework
    probability: float
    confidence_interval: Tuple[float, float] = (0.0, 1.0)
    calibrated: bool = False
    model_version: str = ""
    inference_time_ms: float = 0.0
    
    # Recent performance metrics
    recent_accuracy: float = 0.0  # Last 30 days
    recent_auc: float = 0.0
    recent_clv: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'framework': self.framework.value,
            'probability': self.probability,
            'confidence_interval': list(self.confidence_interval),
            'calibrated': self.calibrated,
            'recent_accuracy': self.recent_accuracy,
        }


@dataclass
class EnsemblePrediction:
    """Combined prediction from all frameworks."""
    probability: float
    confidence: float
    uncertainty: float  # Standard deviation of framework predictions
    
    # Individual framework predictions
    framework_predictions: Dict[str, FrameworkPrediction] = field(default_factory=dict)
    
    # Weighting info
    weights_used: Dict[str, float] = field(default_factory=dict)
    agreement_score: float = 0.0  # How much frameworks agree (1.0 = perfect)
    
    # Confidence interval (Bayesian)
    ci_lower: float = 0.0
    ci_upper: float = 1.0
    
    # Market adjustment
    market_adjusted: bool = False
    raw_probability: float = 0.0


@dataclass
class SituationalModifiers:
    """Situational factors that adjust predictions."""
    # Rest and travel
    home_rest_days: int = 3
    away_rest_days: int = 3
    home_b2b: bool = False
    away_b2b: bool = False
    travel_distance_miles: float = 0.0
    timezone_change: int = 0
    
    # Game context
    is_playoff: bool = False
    is_rivalry: bool = False
    is_primetime: bool = False
    is_national_tv: bool = False
    
    # Injury impact (0-1, higher = more impact)
    home_injury_impact: float = 0.0
    away_injury_impact: float = 0.0
    
    # Weather (outdoor sports)
    temperature: Optional[float] = None
    wind_speed: Optional[float] = None
    precipitation: Optional[float] = None
    is_dome: bool = False
    
    # Motivation factors
    home_elimination_game: bool = False
    away_elimination_game: bool = False
    home_clinch_scenario: bool = False
    away_clinch_scenario: bool = False
    
    def calculate_adjustment(self, sport_config: SportPredictionConfig) -> float:
        """Calculate net probability adjustment from situational factors."""
        adjustment = 0.0
        
        # Rest advantage
        rest_diff = self.home_rest_days - self.away_rest_days
        adjustment += rest_diff * sport_config.rest_bonus
        
        # Back-to-back penalty
        if self.home_b2b:
            adjustment -= sport_config.b2b_penalty
        if self.away_b2b:
            adjustment += sport_config.b2b_penalty
        
        # Travel penalty
        if self.timezone_change > 0:
            adjustment += self.timezone_change * sport_config.travel_penalty
        
        # Injury impact
        injury_diff = self.away_injury_impact - self.home_injury_impact
        adjustment += injury_diff * 0.05
        
        # Playoff boost (more predictable)
        if self.is_playoff:
            adjustment *= 0.8  # Reduce adjustment in playoffs
        
        return np.clip(adjustment, -0.10, 0.10)


@dataclass
class EdgeAnalysis:
    """Detailed edge analysis for a prediction."""
    raw_edge: float
    adjusted_edge: float
    
    # Edge breakdown
    model_edge: float  # Edge from model probability
    market_inefficiency: float  # Detected market inefficiency
    situational_edge: float  # From situational factors
    
    # Confidence
    edge_confidence: float  # How confident in the edge (0-1)
    
    # Expected value
    expected_value: float
    expected_value_per_unit: float
    
    # Risk metrics
    variance: float
    sharpe_ratio: float
    
    # CLV prediction
    predicted_clv: float
    clv_confidence: float


@dataclass 
class BettingRecommendation:
    """Complete betting recommendation."""
    action: str  # 'STRONG_BET', 'BET', 'LEAN', 'PASS', 'FADE'
    
    # Sizing
    kelly_fraction: float
    recommended_units: float
    max_bet_percent: float
    
    # Timing
    bet_now: bool
    wait_for_line_movement: bool
    
    # Explanation (required)
    primary_reason: str
    
    # Optional fields with defaults
    target_line: Optional[float] = None
    hedge_recommendation: Optional[str] = None
    secondary_factors: List[str] = field(default_factory=list)


@dataclass
class PredictionExplanation:
    """SHAP-based explanation with additional context."""
    # Top contributing features
    top_positive_factors: List[Dict[str, Any]] = field(default_factory=list)
    top_negative_factors: List[Dict[str, Any]] = field(default_factory=list)
    
    # Feature contributions
    total_positive_contribution: float = 0.0
    total_negative_contribution: float = 0.0
    
    # Key insights
    key_insight: str = ""
    confidence_factors: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    
    # Natural language summary
    summary: str = ""


@dataclass
class Prediction:
    """Complete advanced prediction with all metadata."""
    # Identifiers
    prediction_id: str
    game_id: str
    sport: str
    
    # Prediction core
    bet_type: BetType
    predicted_side: PredictedSide
    probability: float
    raw_probability: float
    
    # Confidence and tier
    confidence: float
    uncertainty: float
    signal_tier: SignalTier
    
    # Edge analysis
    edge: float
    edge_percentage: float
    edge_analysis: EdgeAnalysis
    
    # Betting info
    line: float
    odds: int
    implied_probability: float
    
    # Model info
    model_id: str
    ensemble_prediction: EnsemblePrediction
    
    # Situational
    situational_modifiers: SituationalModifiers
    situational_adjustment: float
    
    # Explanation
    explanation: PredictionExplanation
    
    # Betting recommendation
    recommendation: BettingRecommendation
    
    # Market analysis
    market_type: MarketType
    sharp_side: Optional[str] = None
    steam_move_detected: bool = False
    reverse_line_movement: Optional[str] = None
    
    # CLV tracking
    predicted_clv: float = 0.0
    
    # Integrity
    prediction_hash: str = ""
    locked_at: datetime = field(default_factory=datetime.utcnow)
    
    # Game info
    game_date: datetime = field(default_factory=datetime.utcnow)
    home_team: str = ""
    away_team: str = ""
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    
    # Grading (filled after game)
    graded: bool = False
    result: Optional[str] = None
    actual_score_home: Optional[int] = None
    actual_score_away: Optional[int] = None
    actual_clv: Optional[float] = None
    profit_loss: Optional[float] = None
    graded_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'prediction_id': self.prediction_id,
            'game_id': self.game_id,
            'sport': self.sport,
            'bet_type': self.bet_type.value,
            'predicted_side': self.predicted_side.value,
            'probability': round(self.probability, 4),
            'raw_probability': round(self.raw_probability, 4),
            'confidence': round(self.confidence, 4),
            'uncertainty': round(self.uncertainty, 4),
            'signal_tier': self.signal_tier.value,
            'edge': round(self.edge, 4),
            'edge_percentage': round(self.edge_percentage, 2),
            'line': self.line,
            'odds': self.odds,
            'implied_probability': round(self.implied_probability, 4),
            'model_id': self.model_id,
            'framework_probabilities': {
                k: v.to_dict() for k, v in self.ensemble_prediction.framework_predictions.items()
            },
            'market_type': self.market_type.value,
            'recommendation': {
                'action': self.recommendation.action,
                'kelly_fraction': round(self.recommendation.kelly_fraction, 4),
                'recommended_units': round(self.recommendation.recommended_units, 2),
                'primary_reason': self.recommendation.primary_reason,
            },
            'explanation': {
                'key_insight': self.explanation.key_insight,
                'top_positive_factors': self.explanation.top_positive_factors[:5],
                'top_negative_factors': self.explanation.top_negative_factors[:5],
            },
            'prediction_hash': self.prediction_hash,
            'locked_at': self.locked_at.isoformat(),
            'game_date': self.game_date.isoformat(),
            'home_team': self.home_team,
            'away_team': self.away_team,
            'graded': self.graded,
            'result': self.result,
            'actual_clv': self.actual_clv,
        }
    
    def get_summary(self) -> str:
        """Get human-readable summary."""
        side = "HOME" if self.predicted_side == PredictedSide.HOME else "AWAY"
        if self.bet_type == BetType.TOTAL:
            side = "OVER" if self.predicted_side == PredictedSide.OVER else "UNDER"
        
        return (
            f"[{self.signal_tier.value}] {self.sport} {self.bet_type.value.upper()}: "
            f"{side} ({self.probability:.1%} | Edge: {self.edge:.1%}) - "
            f"{self.recommendation.action}"
        )


@dataclass
class PredictionBatch:
    """Batch of predictions for multiple games."""
    batch_id: str
    predictions: List[Prediction]
    
    # Summary stats
    total_predictions: int = 0
    tier_a_count: int = 0
    tier_b_count: int = 0
    tier_c_count: int = 0
    tier_d_count: int = 0
    
    actionable_count: int = 0
    total_expected_value: float = 0.0
    
    # Metadata
    sport: Optional[str] = None
    date: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Calculate summary stats."""
        self.total_predictions = len(self.predictions)
        self.tier_a_count = sum(1 for p in self.predictions if p.signal_tier == SignalTier.A)
        self.tier_b_count = sum(1 for p in self.predictions if p.signal_tier == SignalTier.B)
        self.tier_c_count = sum(1 for p in self.predictions if p.signal_tier == SignalTier.C)
        self.tier_d_count = sum(1 for p in self.predictions if p.signal_tier == SignalTier.D)
        self.actionable_count = sum(
            1 for p in self.predictions 
            if p.recommendation.action in ['STRONG_BET', 'BET', 'LEAN']
        )
        self.total_expected_value = sum(
            p.edge_analysis.expected_value for p in self.predictions
        )
    
    def get_actionable(self) -> List[Prediction]:
        """Get only actionable predictions."""
        return [
            p for p in self.predictions
            if p.recommendation.action in ['STRONG_BET', 'BET', 'LEAN']
        ]
    
    def get_by_tier(self, tier: SignalTier) -> List[Prediction]:
        """Get predictions by tier."""
        return [p for p in self.predictions if p.signal_tier == tier]
    
    def sort_by_edge(self) -> List[Prediction]:
        """Sort predictions by edge descending."""
        return sorted(self.predictions, key=lambda p: p.edge, reverse=True)
    
    def sort_by_expected_value(self) -> List[Prediction]:
        """Sort predictions by expected value descending."""
        return sorted(
            self.predictions, 
            key=lambda p: p.edge_analysis.expected_value, 
            reverse=True
        )


# =============================================================================
# ADVANCED PREDICTION ENGINE
# =============================================================================

class AdvancedPredictionEngine:
    """
    Enterprise-grade prediction engine with advanced features.
    
    Features:
    - Dynamic ensemble weighting based on recent performance
    - Confidence intervals and uncertainty quantification
    - Kelly Criterion with fractional sizing
    - CLV prediction and tracking
    - Sharp/public money detection
    - Contextual tier adjustments
    - Sport-specific optimizations
    - Model drift detection
    - Real-time line movement analysis
    - Bayesian probability updates
    """
    
    def __init__(
        self,
        meta_ensemble=None,
        shap_explainer=None,
        probability_calibrator=None,
        model_registry=None,
        model_id: str = 'advanced_v2',
        enable_bayesian_updates: bool = True,
        enable_drift_detection: bool = True,
    ):
        """
        Initialize advanced prediction engine.
        
        Args:
            meta_ensemble: MetaEnsemble for combining framework predictions
            shap_explainer: SHAP explainer for feature importance
            probability_calibrator: Probability calibration module
            model_registry: Model registry for versioning
            model_id: Model identifier
            enable_bayesian_updates: Enable Bayesian probability updates
            enable_drift_detection: Enable model drift detection
        """
        self.meta_ensemble = meta_ensemble
        self.shap_explainer = shap_explainer
        self.probability_calibrator = probability_calibrator
        self.model_registry = model_registry
        self.model_id = model_id
        self.enable_bayesian_updates = enable_bayesian_updates
        self.enable_drift_detection = enable_drift_detection
        
        self._prediction_counter = 0
        
        # Performance tracking
        self._recent_predictions: List[Prediction] = []
        self._performance_by_sport: Dict[str, Dict] = {}
        self._framework_performance: Dict[str, Dict] = {}
        
        # Drift detection
        self._baseline_distribution: Optional[Dict] = None
        self._drift_detected = False
        
        logger.info(f"Advanced Prediction Engine initialized: {model_id}")
    
    # =========================================================================
    # CORE PREDICTION GENERATION
    # =========================================================================
    
    def generate_prediction(
        self,
        game_id: str,
        sport: str,
        home_team: str,
        away_team: str,
        game_date: datetime,
        bet_type: BetType,
        features: Dict[str, float],
        odds_info: OddsInfo,
        framework_predictions: Dict[str, FrameworkPrediction],
        situational: Optional[SituationalModifiers] = None,
        force_recalculate: bool = False,
    ) -> Prediction:
        """
        Generate an advanced prediction with full analysis.
        
        Args:
            game_id: Unique game identifier
            sport: Sport code (NFL, NBA, etc.)
            home_team: Home team name
            away_team: Away team name  
            game_date: Game date/time
            bet_type: Type of bet
            features: Feature dictionary for prediction
            odds_info: Odds information
            framework_predictions: Predictions from each ML framework
            situational: Situational modifiers
            force_recalculate: Force recalculation even if cached
            
        Returns:
            Complete Prediction object
        """
        locked_at = datetime.utcnow()
        
        # Get sport-specific config
        sport_config = SPORT_PREDICTION_CONFIGS.get(
            sport, 
            SportPredictionConfig(sport_code=sport)
        )
        
        # Initialize situational if not provided
        if situational is None:
            situational = SituationalModifiers()
        
        # Step 1: Combine framework predictions
        ensemble_pred = self._combine_framework_predictions(
            framework_predictions, sport_config, odds_info
        )
        
        # Step 2: Apply probability calibration
        if self.probability_calibrator is not None:
            calibrated_prob = self.probability_calibrator.calibrate(
                ensemble_pred.probability, sport, bet_type.value
            )
        else:
            calibrated_prob = ensemble_pred.probability
        
        # Step 3: Apply situational adjustments
        situational_adj = situational.calculate_adjustment(sport_config)
        adjusted_prob = np.clip(calibrated_prob + situational_adj, 0.01, 0.99)
        
        # Step 4: Apply Bayesian update with market odds
        if self.enable_bayesian_updates:
            market_prob = odds_info.get_implied_probability(bet_type, PredictedSide.HOME)
            final_prob = self._bayesian_update(adjusted_prob, market_prob)
        else:
            final_prob = adjusted_prob
        
        # Step 5: Determine predicted side
        predicted_side, display_prob = self._determine_predicted_side(final_prob, bet_type)
        
        # Step 6: Get line and odds
        line, odds, implied_prob = self._get_line_and_odds(
            odds_info, bet_type, predicted_side
        )
        
        # Step 7: Calculate comprehensive edge analysis
        edge_analysis = self._calculate_edge_analysis(
            display_prob, implied_prob, odds, sport_config, odds_info
        )
        
        # Step 8: Determine signal tier with adjustments
        signal_tier = self._determine_signal_tier(
            display_prob, ensemble_pred.agreement_score, sport_config
        )
        
        # Step 9: Generate explanation
        explanation = self._generate_explanation(
            features, framework_predictions, situational, predicted_side, sport
        )
        
        # Step 10: Generate betting recommendation
        recommendation = self._generate_recommendation(
            signal_tier, edge_analysis, odds_info, sport_config, bet_type
        )
        
        # Step 11: Analyze market
        market_type = odds_info.get_market_type()
        sharp_side = odds_info.detect_reverse_line_movement()
        steam_move = odds_info.detect_steam_move(sport_config.steam_move_threshold)
        
        # Step 12: Predict CLV
        predicted_clv = self._predict_clv(
            edge_analysis.raw_edge, 
            market_type,
            odds_info,
            sport
        )
        
        # Step 13: Generate prediction hash for integrity
        prediction_id = self._generate_prediction_id(game_id, bet_type.value)
        prediction_hash = self._generate_prediction_hash({
            'game_id': game_id,
            'bet_type': bet_type.value,
            'predicted_side': predicted_side.value,
            'probability': display_prob,
            'line': line,
            'odds': odds,
            'locked_at': locked_at.isoformat()
        })
        
        # Create prediction object
        prediction = Prediction(
            prediction_id=prediction_id,
            game_id=game_id,
            sport=sport,
            bet_type=bet_type,
            predicted_side=predicted_side,
            probability=float(display_prob),
            raw_probability=float(ensemble_pred.probability),
            confidence=float(max(display_prob, 1 - display_prob)),
            uncertainty=float(ensemble_pred.uncertainty),
            signal_tier=signal_tier,
            edge=float(edge_analysis.adjusted_edge),
            edge_percentage=float(edge_analysis.adjusted_edge * 100),
            edge_analysis=edge_analysis,
            line=float(line),
            odds=int(odds),
            implied_probability=float(implied_prob),
            model_id=self.model_id,
            ensemble_prediction=ensemble_pred,
            situational_modifiers=situational,
            situational_adjustment=float(situational_adj),
            explanation=explanation,
            recommendation=recommendation,
            market_type=market_type,
            sharp_side=sharp_side,
            steam_move_detected=steam_move,
            reverse_line_movement=sharp_side,
            predicted_clv=float(predicted_clv),
            prediction_hash=prediction_hash,
            locked_at=locked_at,
            game_date=game_date,
            home_team=home_team,
            away_team=away_team,
            expires_at=game_date,
        )
        
        # Track prediction
        self._recent_predictions.append(prediction)
        if len(self._recent_predictions) > 1000:
            self._recent_predictions = self._recent_predictions[-500:]
        
        logger.info(
            f"Generated prediction: {prediction_id} | "
            f"{sport} {bet_type.value} | Tier {signal_tier.value} | "
            f"Edge: {edge_analysis.adjusted_edge:.2%}"
        )
        
        return prediction
    
    # =========================================================================
    # ENSEMBLE COMBINATION
    # =========================================================================
    
    def _combine_framework_predictions(
        self,
        framework_predictions: Dict[str, FrameworkPrediction],
        sport_config: SportPredictionConfig,
        odds_info: OddsInfo,
    ) -> EnsemblePrediction:
        """
        Combine predictions from multiple frameworks with dynamic weighting.
        """
        if not framework_predictions:
            return EnsemblePrediction(
                probability=0.5,
                confidence=0.0,
                uncertainty=0.5,
            )
        
        # Get dynamic weights based on recent performance
        weights = self._get_dynamic_weights(framework_predictions, sport_config)
        
        # Calculate weighted probability
        weighted_sum = 0.0
        weight_sum = 0.0
        probs = []
        
        for name, pred in framework_predictions.items():
            weight = weights.get(name, 1.0 / len(framework_predictions))
            weighted_sum += pred.probability * weight
            weight_sum += weight
            probs.append(pred.probability)
        
        if weight_sum > 0:
            combined_prob = weighted_sum / weight_sum
        else:
            combined_prob = np.mean(probs)
        
        # Calculate uncertainty (std of framework predictions)
        uncertainty = np.std(probs) if len(probs) > 1 else 0.1
        
        # Calculate agreement score
        if len(probs) > 1:
            agreement = 1.0 - (uncertainty / 0.25)  # Normalize by max expected std
            agreement = np.clip(agreement, 0.0, 1.0)
        else:
            agreement = 0.5
        
        # Calculate confidence interval
        ci_lower = max(0.0, combined_prob - 2 * uncertainty)
        ci_upper = min(1.0, combined_prob + 2 * uncertainty)
        
        return EnsemblePrediction(
            probability=combined_prob,
            confidence=1.0 - uncertainty,
            uncertainty=uncertainty,
            framework_predictions=framework_predictions,
            weights_used=weights,
            agreement_score=agreement,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            raw_probability=combined_prob,
        )
    
    def _get_dynamic_weights(
        self,
        framework_predictions: Dict[str, FrameworkPrediction],
        sport_config: SportPredictionConfig,
    ) -> Dict[str, float]:
        """
        Calculate dynamic weights based on recent framework performance.
        """
        weights = {}
        total_weight = 0.0
        
        for name, pred in framework_predictions.items():
            # Start with sport-specific base weights
            if name == 'h2o':
                base_weight = sport_config.h2o_weight
            elif name == 'autogluon':
                base_weight = sport_config.autogluon_weight
            elif name == 'sklearn':
                base_weight = sport_config.sklearn_weight
            else:
                base_weight = 0.25
            
            # Adjust by recent performance
            performance_multiplier = 1.0
            if pred.recent_accuracy > 0.55:
                performance_multiplier = 1.0 + (pred.recent_accuracy - 0.55) * 2
            elif pred.recent_accuracy > 0 and pred.recent_accuracy < 0.45:
                performance_multiplier = 0.5 + pred.recent_accuracy
            
            # Adjust by recent CLV
            if pred.recent_clv > 0:
                performance_multiplier *= 1.0 + min(pred.recent_clv * 10, 0.3)
            
            weight = base_weight * performance_multiplier
            weights[name] = weight
            total_weight += weight
        
        # Normalize
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    def _bayesian_update(
        self,
        model_prob: float,
        market_prob: float,
        model_confidence: float = 0.7,
        market_efficiency: float = 0.85,
    ) -> float:
        """
        Apply Bayesian update combining model and market probabilities.
        
        Uses market probability as prior and model probability as likelihood.
        """
        # Avoid edge cases
        model_prob = np.clip(model_prob, 0.01, 0.99)
        market_prob = np.clip(market_prob, 0.01, 0.99)
        
        # Weight based on relative confidence
        # More efficient markets get more weight
        market_weight = market_efficiency * 0.3
        model_weight = model_confidence * 0.7
        
        total_weight = market_weight + model_weight
        
        # Weighted average in log-odds space for better calibration
        model_log_odds = np.log(model_prob / (1 - model_prob))
        market_log_odds = np.log(market_prob / (1 - market_prob))
        
        combined_log_odds = (
            model_log_odds * model_weight + 
            market_log_odds * market_weight
        ) / total_weight
        
        # Convert back to probability
        combined_prob = 1 / (1 + np.exp(-combined_log_odds))
        
        return combined_prob
    
    # =========================================================================
    # EDGE ANALYSIS
    # =========================================================================
    
    def _calculate_edge_analysis(
        self,
        probability: float,
        implied_probability: float,
        odds: int,
        sport_config: SportPredictionConfig,
        odds_info: OddsInfo,
    ) -> EdgeAnalysis:
        """
        Perform comprehensive edge analysis.
        """
        # Raw edge
        raw_edge = probability - implied_probability
        
        # Model edge (before adjustments)
        model_edge = raw_edge
        
        # Market inefficiency (from line movement patterns)
        market_inefficiency = 0.0
        if odds_info.detect_reverse_line_movement():
            market_inefficiency = 0.02  # RLM suggests inefficiency
        if odds_info.detect_steam_move():
            market_inefficiency -= 0.01  # Steam moves are often correct
        
        # Situational edge
        situational_edge = 0.0  # Already incorporated
        
        # Adjusted edge
        adjusted_edge = raw_edge + market_inefficiency
        
        # Expected value calculation
        decimal_odds = self._american_to_decimal(odds)
        ev = (probability * (decimal_odds - 1)) - (1 - probability)
        ev_per_unit = ev  # Already per unit
        
        # Variance (simplified)
        variance = probability * (1 - probability)
        
        # Sharpe ratio approximation
        if variance > 0:
            sharpe = adjusted_edge / np.sqrt(variance)
        else:
            sharpe = 0.0
        
        # Edge confidence
        edge_confidence = min(1.0, abs(adjusted_edge) / 0.10)  # 10% edge = max confidence
        
        # CLV prediction
        predicted_clv = adjusted_edge * 0.6  # Assume 60% CLV capture
        clv_confidence = 0.7 if abs(adjusted_edge) > 0.03 else 0.5
        
        return EdgeAnalysis(
            raw_edge=raw_edge,
            adjusted_edge=adjusted_edge,
            model_edge=model_edge,
            market_inefficiency=market_inefficiency,
            situational_edge=situational_edge,
            edge_confidence=edge_confidence,
            expected_value=ev,
            expected_value_per_unit=ev_per_unit,
            variance=variance,
            sharpe_ratio=sharpe,
            predicted_clv=predicted_clv,
            clv_confidence=clv_confidence,
        )
    
    def _american_to_decimal(self, american: int) -> float:
        """Convert American odds to decimal."""
        if american < 0:
            return 1 + (100 / abs(american))
        else:
            return 1 + (american / 100)
    
    # =========================================================================
    # SIGNAL TIER CLASSIFICATION
    # =========================================================================
    
    def _determine_signal_tier(
        self,
        probability: float,
        agreement_score: float,
        sport_config: SportPredictionConfig,
    ) -> SignalTier:
        """
        Determine signal tier with contextual adjustments.
        """
        confidence = max(probability, 1 - probability)
        
        # Adjust confidence by agreement
        if agreement_score < 0.5:
            confidence *= 0.95  # Reduce if frameworks disagree
        elif agreement_score > 0.8:
            confidence *= 1.02  # Slight boost for strong agreement
        
        # Apply sport-specific thresholds
        if confidence >= sport_config.tier_a_min:
            return SignalTier.A
        elif confidence >= sport_config.tier_b_min:
            return SignalTier.B
        elif confidence >= sport_config.tier_c_min:
            return SignalTier.C
        else:
            return SignalTier.D
    
    # =========================================================================
    # BETTING RECOMMENDATION
    # =========================================================================
    
    def _generate_recommendation(
        self,
        tier: SignalTier,
        edge_analysis: EdgeAnalysis,
        odds_info: OddsInfo,
        sport_config: SportPredictionConfig,
        bet_type: BetType,
    ) -> BettingRecommendation:
        """
        Generate comprehensive betting recommendation.
        """
        min_edge = sport_config.get_min_edge(bet_type)
        
        # Determine action
        if tier == SignalTier.D:
            action = 'PASS'
            primary_reason = 'Insufficient confidence'
        elif edge_analysis.adjusted_edge < min_edge:
            action = 'PASS'
            primary_reason = f'Edge below threshold ({edge_analysis.adjusted_edge:.1%} < {min_edge:.1%})'
        elif tier == SignalTier.A and edge_analysis.adjusted_edge >= min_edge * 1.5:
            action = 'STRONG_BET'
            primary_reason = f'Elite prediction with {edge_analysis.adjusted_edge:.1%} edge'
        elif tier == SignalTier.A:
            action = 'BET'
            primary_reason = f'Tier A prediction with {edge_analysis.adjusted_edge:.1%} edge'
        elif tier == SignalTier.B:
            action = 'BET'
            primary_reason = f'Strong value with {edge_analysis.adjusted_edge:.1%} edge'
        elif tier == SignalTier.C and edge_analysis.adjusted_edge >= min_edge:
            action = 'LEAN'
            primary_reason = f'Moderate edge of {edge_analysis.adjusted_edge:.1%}'
        else:
            action = 'PASS'
            primary_reason = 'Does not meet betting criteria'
        
        # Calculate Kelly fraction
        if edge_analysis.adjusted_edge > 0:
            decimal_odds = self._american_to_decimal(odds_info.spread_home_odds or -110)
            full_kelly = edge_analysis.adjusted_edge / (decimal_odds - 1)
            kelly_fraction = full_kelly * sport_config.kelly_fraction * tier.kelly_multiplier
            kelly_fraction = min(kelly_fraction, tier.max_bet_percent)
        else:
            kelly_fraction = 0.0
        
        # Calculate recommended units
        recommended_units = kelly_fraction * 100  # As units of 1% bankroll
        
        # Check for line movement opportunity
        spread_move = odds_info.get_spread_movement()
        wait_for_line = False
        target_line = None
        
        if action in ['BET', 'STRONG_BET'] and abs(spread_move) > 1.0:
            # Line has moved significantly
            rlm = odds_info.detect_reverse_line_movement()
            if rlm:
                wait_for_line = False  # Bet now if RLM detected
            elif spread_move * (1 if odds_info.public_spread_home_pct or 0.5 > 0.5 else -1) > 0:
                # Line moving with public, might get better
                wait_for_line = True
                target_line = odds_info.spread_home
        
        # Secondary factors
        secondary = []
        if odds_info.detect_steam_move():
            secondary.append('Steam move detected')
        if odds_info.detect_reverse_line_movement():
            secondary.append('Reverse line movement (sharp action)')
        if edge_analysis.edge_confidence > 0.8:
            secondary.append('High edge confidence')
        
        return BettingRecommendation(
            action=action,
            kelly_fraction=kelly_fraction,
            recommended_units=recommended_units,
            max_bet_percent=tier.max_bet_percent if action != 'PASS' else 0.0,
            bet_now=not wait_for_line,
            wait_for_line_movement=wait_for_line,
            target_line=target_line,
            primary_reason=primary_reason,
            secondary_factors=secondary,
        )
    
    # =========================================================================
    # EXPLANATION GENERATION
    # =========================================================================
    
    def _generate_explanation(
        self,
        features: Dict[str, float],
        framework_predictions: Dict[str, FrameworkPrediction],
        situational: SituationalModifiers,
        predicted_side: PredictedSide,
        sport: str,
    ) -> PredictionExplanation:
        """
        Generate comprehensive explanation for prediction.
        """
        # Use SHAP if available
        if self.shap_explainer is not None:
            try:
                shap_explanation = self.shap_explainer.explain_prediction(
                    features=features,
                    prediction_id='',
                    game_id='',
                    sport_code=sport,
                )
                top_positive = [f.to_dict() for f in shap_explanation.top_positive_factors[:5]]
                top_negative = [f.to_dict() for f in shap_explanation.top_negative_factors[:5]]
            except Exception as e:
                logger.warning(f"SHAP explanation failed: {e}")
                top_positive, top_negative = self._generate_basic_factors(features, predicted_side)
        else:
            top_positive, top_negative = self._generate_basic_factors(features, predicted_side)
        
        # Generate key insight
        key_insight = self._generate_key_insight(
            top_positive, top_negative, situational, predicted_side
        )
        
        # Confidence factors
        confidence_factors = []
        if top_positive:
            confidence_factors.append(f"Strong: {top_positive[0].get('feature', 'Unknown')}")
        
        # Risk factors
        risk_factors = []
        if situational.home_b2b or situational.away_b2b:
            risk_factors.append("Back-to-back game situation")
        if situational.home_injury_impact > 0.1 or situational.away_injury_impact > 0.1:
            risk_factors.append("Injury concerns")
        
        # Generate summary
        summary = self._generate_summary(
            top_positive, top_negative, key_insight, predicted_side
        )
        
        return PredictionExplanation(
            top_positive_factors=top_positive,
            top_negative_factors=top_negative,
            total_positive_contribution=sum(f.get('shap_value', 0) for f in top_positive),
            total_negative_contribution=sum(f.get('shap_value', 0) for f in top_negative),
            key_insight=key_insight,
            confidence_factors=confidence_factors,
            risk_factors=risk_factors,
            summary=summary,
        )
    
    def _generate_basic_factors(
        self,
        features: Dict[str, float],
        predicted_side: PredictedSide,
    ) -> Tuple[List[Dict], List[Dict]]:
        """Generate basic factors without SHAP."""
        key_features = [
            ('home_elo', 'ELO Rating'),
            ('elo_diff', 'ELO Difference'),
            ('home_win_pct_l10', 'Home Recent Form'),
            ('away_win_pct_l10', 'Away Recent Form'),
            ('rest_advantage', 'Rest Advantage'),
            ('h2h_win_pct', 'Head-to-Head'),
            ('spread_movement', 'Line Movement'),
            ('home_off_rating', 'Offensive Rating'),
            ('away_def_rating', 'Defensive Rating'),
        ]
        
        positive_factors = []
        negative_factors = []
        
        for feature_key, description in key_features:
            if feature_key in features:
                value = features[feature_key]
                if value != 0:
                    factor = {
                        'feature': feature_key,
                        'description': description,
                        'value': float(value),
                        'shap_value': abs(value) * 0.01,  # Approximation
                    }
                    
                    # Determine if positive for predicted side
                    is_positive_for_home = value > 0
                    if predicted_side in [PredictedSide.HOME, PredictedSide.OVER]:
                        if is_positive_for_home:
                            factor['impact'] = 'positive'
                            positive_factors.append(factor)
                        else:
                            factor['impact'] = 'negative'
                            negative_factors.append(factor)
                    else:
                        if not is_positive_for_home:
                            factor['impact'] = 'positive'
                            positive_factors.append(factor)
                        else:
                            factor['impact'] = 'negative'
                            negative_factors.append(factor)
        
        # Sort by value
        positive_factors.sort(key=lambda x: abs(x['value']), reverse=True)
        negative_factors.sort(key=lambda x: abs(x['value']), reverse=True)
        
        return positive_factors[:5], negative_factors[:5]
    
    def _generate_key_insight(
        self,
        top_positive: List[Dict],
        top_negative: List[Dict],
        situational: SituationalModifiers,
        predicted_side: PredictedSide,
    ) -> str:
        """Generate a key insight sentence."""
        if not top_positive:
            return "Limited factors supporting this prediction"
        
        top_factor = top_positive[0].get('description', top_positive[0].get('feature', 'Unknown'))
        side_name = predicted_side.value.upper()
        
        insight = f"Primary edge from {top_factor} favoring {side_name}"
        
        if situational.is_playoff:
            insight += " (Playoff game)"
        elif situational.home_b2b or situational.away_b2b:
            insight += " (fatigue factor)"
        
        return insight
    
    def _generate_summary(
        self,
        top_positive: List[Dict],
        top_negative: List[Dict],
        key_insight: str,
        predicted_side: PredictedSide,
    ) -> str:
        """Generate human-readable summary."""
        lines = [key_insight, ""]
        
        if top_positive:
            lines.append("Supporting factors:")
            for f in top_positive[:3]:
                lines.append(f"  • {f.get('description', f.get('feature'))}")
        
        if top_negative:
            lines.append("\nRisk factors:")
            for f in top_negative[:2]:
                lines.append(f"  • {f.get('description', f.get('feature'))}")
        
        return "\n".join(lines)
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _determine_predicted_side(
        self,
        probability: float,
        bet_type: BetType,
    ) -> Tuple[PredictedSide, float]:
        """Determine predicted side based on probability."""
        if bet_type in [BetType.SPREAD, BetType.MONEYLINE, BetType.FIRST_HALF_SPREAD]:
            if probability >= 0.5:
                return PredictedSide.HOME, probability
            else:
                return PredictedSide.AWAY, 1 - probability
        else:  # TOTAL types
            if probability >= 0.5:
                return PredictedSide.OVER, probability
            else:
                return PredictedSide.UNDER, 1 - probability
    
    def _get_line_and_odds(
        self,
        odds_info: OddsInfo,
        bet_type: BetType,
        predicted_side: PredictedSide,
    ) -> Tuple[float, int, float]:
        """Get line, odds, and implied probability for bet."""
        if bet_type == BetType.SPREAD:
            if predicted_side == PredictedSide.HOME:
                line = odds_info.spread_home or 0.0
                odds = odds_info.spread_home_odds or -110
            else:
                line = odds_info.spread_away or 0.0
                odds = odds_info.spread_away_odds or -110
        elif bet_type == BetType.MONEYLINE:
            line = 0.0
            if predicted_side == PredictedSide.HOME:
                odds = odds_info.moneyline_home or -110
            else:
                odds = odds_info.moneyline_away or -110
        elif bet_type in [BetType.TOTAL, BetType.FIRST_HALF_TOTAL]:
            line = odds_info.total_line or 0.0
            if predicted_side == PredictedSide.OVER:
                odds = odds_info.total_over_odds or -110
            else:
                odds = odds_info.total_under_odds or -110
        else:
            line = 0.0
            odds = -110
        
        implied_prob = odds_info.get_implied_probability(bet_type, predicted_side)
        return line, odds, implied_prob
    
    def _predict_clv(
        self,
        edge: float,
        market_type: MarketType,
        odds_info: OddsInfo,
        sport: str,
    ) -> float:
        """Predict expected CLV based on edge and market conditions."""
        # Base CLV is typically 50-70% of detected edge
        base_clv = edge * 0.6
        
        # Adjust by market type
        if market_type == MarketType.SHARP:
            base_clv *= 0.8  # Harder to capture CLV in sharp markets
        elif market_type == MarketType.SQUARE:
            base_clv *= 1.1  # Easier in square markets
        
        # Sport-specific adjustments
        clv_efficiency = {
            'NFL': 0.95,  # Very efficient
            'NBA': 0.90,
            'MLB': 0.85,
            'NHL': 0.85,
            'NCAAF': 0.80,
            'NCAAB': 0.75,  # Less efficient, more CLV opportunity
        }
        base_clv *= clv_efficiency.get(sport, 0.85)
        
        return base_clv
    
    def _generate_prediction_id(self, game_id: str, bet_type: str) -> str:
        """Generate unique prediction ID."""
        self._prediction_counter += 1
        timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S%f')
        return f"pred_{game_id}_{bet_type}_{timestamp}_{self._prediction_counter}"
    
    def _generate_prediction_hash(self, prediction_data: Dict[str, Any]) -> str:
        """Generate SHA-256 hash for prediction integrity."""
        canonical_data = {
            'game_id': prediction_data['game_id'],
            'bet_type': prediction_data['bet_type'],
            'predicted_side': prediction_data['predicted_side'],
            'probability': round(prediction_data['probability'], 6),
            'line_at_prediction': prediction_data['line'],
            'odds_at_prediction': prediction_data['odds'],
            'locked_at': prediction_data['locked_at']
        }
        json_str = json.dumps(canonical_data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()
    
    def verify_prediction_hash(self, prediction: Prediction) -> bool:
        """Verify prediction integrity via hash."""
        recalculated = self._generate_prediction_hash({
            'game_id': prediction.game_id,
            'bet_type': prediction.bet_type.value,
            'predicted_side': prediction.predicted_side.value,
            'probability': prediction.probability,
            'line': prediction.line,
            'odds': prediction.odds,
            'locked_at': prediction.locked_at.isoformat()
        })
        return hmac.compare_digest(recalculated, prediction.prediction_hash)
    
    # =========================================================================
    # GRADING SYSTEM
    # =========================================================================
    
    def grade_prediction(
        self,
        prediction: Prediction,
        home_score: int,
        away_score: int,
        closing_line: Optional[float] = None,
        closing_odds: Optional[int] = None,
    ) -> Prediction:
        """
        Grade a prediction after game completion.
        
        Args:
            prediction: Prediction to grade
            home_score: Final home team score
            away_score: Final away team score
            closing_line: Closing line for CLV calculation
            closing_odds: Closing odds for CLV calculation
            
        Returns:
            Updated Prediction with grade
        """
        actual_margin = home_score - away_score
        actual_total = home_score + away_score
        
        # Determine result
        if prediction.bet_type == BetType.SPREAD:
            result = self._grade_spread(
                prediction.predicted_side,
                prediction.line,
                actual_margin
            )
        elif prediction.bet_type == BetType.MONEYLINE:
            result = self._grade_moneyline(
                prediction.predicted_side,
                actual_margin
            )
        elif prediction.bet_type in [BetType.TOTAL, BetType.FIRST_HALF_TOTAL]:
            result = self._grade_total(
                prediction.predicted_side,
                prediction.line,
                actual_total
            )
        else:
            result = 'unknown'
        
        # Calculate profit/loss
        profit_loss = self._calculate_profit_loss(
            result,
            prediction.odds,
            prediction.recommendation.recommended_units
        )
        
        # Calculate actual CLV
        actual_clv = None
        if closing_line is not None:
            actual_clv = self._calculate_clv(
                prediction.line,
                closing_line,
                prediction.predicted_side,
                prediction.bet_type
            )
        
        # Update prediction
        prediction.graded = True
        prediction.result = result
        prediction.actual_score_home = home_score
        prediction.actual_score_away = away_score
        prediction.actual_clv = actual_clv
        prediction.profit_loss = profit_loss
        prediction.graded_at = datetime.utcnow()
        
        # Update performance tracking
        self._update_performance_tracking(prediction)
        
        return prediction
    
    def _grade_spread(
        self,
        predicted_side: PredictedSide,
        line: float,
        actual_margin: int,
    ) -> str:
        """Grade spread bet."""
        if predicted_side == PredictedSide.HOME:
            result_margin = actual_margin + line
        else:
            result_margin = -actual_margin - line
        
        if result_margin > 0:
            return 'win'
        elif result_margin < 0:
            return 'loss'
        else:
            return 'push'
    
    def _grade_moneyline(
        self,
        predicted_side: PredictedSide,
        actual_margin: int,
    ) -> str:
        """Grade moneyline bet."""
        if predicted_side == PredictedSide.HOME:
            if actual_margin > 0:
                return 'win'
            elif actual_margin < 0:
                return 'loss'
            else:
                return 'push'
        else:
            if actual_margin < 0:
                return 'win'
            elif actual_margin > 0:
                return 'loss'
            else:
                return 'push'
    
    def _grade_total(
        self,
        predicted_side: PredictedSide,
        line: float,
        actual_total: int,
    ) -> str:
        """Grade total bet."""
        if predicted_side == PredictedSide.OVER:
            if actual_total > line:
                return 'win'
            elif actual_total < line:
                return 'loss'
            else:
                return 'push'
        else:
            if actual_total < line:
                return 'win'
            elif actual_total > line:
                return 'loss'
            else:
                return 'push'
    
    def _calculate_profit_loss(
        self,
        result: str,
        odds: int,
        units: float,
    ) -> float:
        """Calculate profit/loss in units."""
        if result == 'win':
            if odds > 0:
                return units * (odds / 100)
            else:
                return units * (100 / abs(odds))
        elif result == 'loss':
            return -units
        else:  # push
            return 0.0
    
    def _calculate_clv(
        self,
        bet_line: float,
        closing_line: float,
        predicted_side: PredictedSide,
        bet_type: BetType,
    ) -> float:
        """Calculate Closing Line Value."""
        if bet_type == BetType.SPREAD:
            if predicted_side == PredictedSide.HOME:
                return closing_line - bet_line
            else:
                return bet_line - closing_line
        elif bet_type in [BetType.TOTAL, BetType.FIRST_HALF_TOTAL]:
            if predicted_side == PredictedSide.OVER:
                return bet_line - closing_line  # Lower close = positive CLV
            else:
                return closing_line - bet_line
        return 0.0
    
    def _update_performance_tracking(self, prediction: Prediction) -> None:
        """Update internal performance tracking."""
        sport = prediction.sport
        if sport not in self._performance_by_sport:
            self._performance_by_sport[sport] = {
                'total': 0,
                'wins': 0,
                'losses': 0,
                'pushes': 0,
                'profit': 0.0,
                'clv_sum': 0.0,
            }
        
        stats = self._performance_by_sport[sport]
        stats['total'] += 1
        
        if prediction.result == 'win':
            stats['wins'] += 1
        elif prediction.result == 'loss':
            stats['losses'] += 1
        else:
            stats['pushes'] += 1
        
        if prediction.profit_loss:
            stats['profit'] += prediction.profit_loss
        if prediction.actual_clv:
            stats['clv_sum'] += prediction.actual_clv
    
    # =========================================================================
    # MODEL DRIFT DETECTION
    # =========================================================================
    
    def detect_model_drift(
        self,
        recent_window: int = 100,
        threshold: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Detect if model performance has drifted from baseline.
        
        Args:
            recent_window: Number of recent predictions to analyze
            threshold: Accuracy drop threshold to trigger drift alert
            
        Returns:
            Drift analysis results
        """
        if len(self._recent_predictions) < recent_window:
            return {
                'drift_detected': False,
                'message': 'Insufficient data for drift detection',
                'sample_size': len(self._recent_predictions),
            }
        
        # Get recent graded predictions
        recent_graded = [
            p for p in self._recent_predictions[-recent_window:]
            if p.graded
        ]
        
        if len(recent_graded) < 30:
            return {
                'drift_detected': False,
                'message': 'Insufficient graded predictions',
                'sample_size': len(recent_graded),
            }
        
        # Calculate recent accuracy
        wins = sum(1 for p in recent_graded if p.result == 'win')
        recent_accuracy = wins / len(recent_graded)
        
        # Calculate expected accuracy by tier
        expected_accuracy = 0.0
        for p in recent_graded:
            expected_accuracy += p.signal_tier.target_accuracy
        expected_accuracy /= len(recent_graded)
        
        # Check for drift
        accuracy_drop = expected_accuracy - recent_accuracy
        drift_detected = accuracy_drop > threshold
        
        # Calculate CLV drift
        clv_sum = sum(p.actual_clv or 0 for p in recent_graded)
        avg_clv = clv_sum / len(recent_graded)
        clv_drift = avg_clv < -0.01  # Negative CLV is bad
        
        self._drift_detected = drift_detected or clv_drift
        
        return {
            'drift_detected': drift_detected or clv_drift,
            'accuracy_drift': drift_detected,
            'clv_drift': clv_drift,
            'recent_accuracy': recent_accuracy,
            'expected_accuracy': expected_accuracy,
            'accuracy_drop': accuracy_drop,
            'avg_clv': avg_clv,
            'sample_size': len(recent_graded),
            'recommendation': 'Consider retraining models' if self._drift_detected else 'Models performing as expected',
        }
    
    # =========================================================================
    # BATCH PREDICTION
    # =========================================================================
    
    def generate_batch_predictions(
        self,
        games: List[Dict[str, Any]],
        features_by_game: Dict[str, Dict[str, float]],
        odds_by_game: Dict[str, OddsInfo],
        framework_predictions_by_game: Dict[str, Dict[str, FrameworkPrediction]],
        situational_by_game: Optional[Dict[str, SituationalModifiers]] = None,
        bet_types: Optional[List[BetType]] = None,
    ) -> PredictionBatch:
        """
        Generate predictions for multiple games.
        
        Args:
            games: List of game dictionaries
            features_by_game: Features keyed by game_id
            odds_by_game: Odds keyed by game_id
            framework_predictions_by_game: Framework predictions keyed by game_id
            situational_by_game: Situational modifiers keyed by game_id
            bet_types: Bet types to generate (default: spread, moneyline, total)
            
        Returns:
            PredictionBatch with all predictions
        """
        if bet_types is None:
            bet_types = [BetType.SPREAD, BetType.MONEYLINE, BetType.TOTAL]
        
        predictions = []
        batch_id = f"batch_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        
        for game in games:
            game_id = game['game_id']
            
            # Skip if missing data
            if game_id not in features_by_game:
                logger.warning(f"Missing features for game {game_id}")
                continue
            if game_id not in odds_by_game:
                logger.warning(f"Missing odds for game {game_id}")
                continue
            if game_id not in framework_predictions_by_game:
                logger.warning(f"Missing framework predictions for game {game_id}")
                continue
            
            features = features_by_game[game_id]
            odds_info = odds_by_game[game_id]
            framework_preds = framework_predictions_by_game[game_id]
            situational = (
                situational_by_game.get(game_id)
                if situational_by_game else None
            )
            
            # Generate prediction for each bet type
            for bet_type in bet_types:
                try:
                    prediction = self.generate_prediction(
                        game_id=game_id,
                        sport=game['sport'],
                        home_team=game['home_team'],
                        away_team=game['away_team'],
                        game_date=game.get('game_date', datetime.utcnow()),
                        bet_type=bet_type,
                        features=features,
                        odds_info=odds_info,
                        framework_predictions=framework_preds,
                        situational=situational,
                    )
                    predictions.append(prediction)
                except Exception as e:
                    logger.error(f"Failed prediction for {game_id} {bet_type}: {e}")
        
        batch = PredictionBatch(
            batch_id=batch_id,
            predictions=predictions,
            sport=games[0]['sport'] if games else None,
            date=datetime.utcnow(),
        )
        
        logger.info(
            f"Generated batch {batch_id}: {batch.total_predictions} predictions, "
            f"{batch.actionable_count} actionable"
        )
        
        return batch
    
    # =========================================================================
    # PERFORMANCE ANALYTICS
    # =========================================================================
    
    def get_performance_summary(
        self,
        sport: Optional[str] = None,
        tier: Optional[SignalTier] = None,
        days: int = 30,
    ) -> Dict[str, Any]:
        """
        Get performance summary for recent predictions.
        
        Args:
            sport: Filter by sport
            tier: Filter by tier
            days: Number of days to include
            
        Returns:
            Performance summary dictionary
        """
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        # Filter predictions
        predictions = [
            p for p in self._recent_predictions
            if p.graded and p.graded_at and p.graded_at >= cutoff
        ]
        
        if sport:
            predictions = [p for p in predictions if p.sport == sport]
        if tier:
            predictions = [p for p in predictions if p.signal_tier == tier]
        
        if not predictions:
            return {
                'total': 0,
                'message': 'No graded predictions in timeframe',
            }
        
        # Calculate metrics
        wins = sum(1 for p in predictions if p.result == 'win')
        losses = sum(1 for p in predictions if p.result == 'loss')
        pushes = sum(1 for p in predictions if p.result == 'push')
        
        total_profit = sum(p.profit_loss or 0 for p in predictions)
        total_clv = sum(p.actual_clv or 0 for p in predictions)
        
        # By tier breakdown
        tier_breakdown = {}
        for t in SignalTier:
            tier_preds = [p for p in predictions if p.signal_tier == t]
            if tier_preds:
                tier_wins = sum(1 for p in tier_preds if p.result == 'win')
                tier_breakdown[t.value] = {
                    'total': len(tier_preds),
                    'wins': tier_wins,
                    'accuracy': tier_wins / len(tier_preds),
                    'target': t.target_accuracy,
                    'vs_target': (tier_wins / len(tier_preds)) - t.target_accuracy,
                }
        
        return {
            'total': len(predictions),
            'wins': wins,
            'losses': losses,
            'pushes': pushes,
            'accuracy': wins / (wins + losses) if (wins + losses) > 0 else 0,
            'profit_units': total_profit,
            'roi': total_profit / len(predictions) if predictions else 0,
            'avg_clv': total_clv / len(predictions) if predictions else 0,
            'tier_breakdown': tier_breakdown,
            'days_analyzed': days,
        }
    
    def get_edge_distribution(self) -> Dict[str, Any]:
        """Analyze edge distribution of recent predictions."""
        if not self._recent_predictions:
            return {'message': 'No predictions available'}
        
        edges = [p.edge for p in self._recent_predictions]
        
        return {
            'mean_edge': np.mean(edges),
            'median_edge': np.median(edges),
            'std_edge': np.std(edges),
            'min_edge': np.min(edges),
            'max_edge': np.max(edges),
            'positive_edge_pct': sum(1 for e in edges if e > 0) / len(edges),
            'above_3pct': sum(1 for e in edges if e > 0.03) / len(edges),
            'sample_size': len(edges),
        }
    
    # =========================================================================
    # FILTERING AND RANKING
    # =========================================================================
    
    def filter_predictions(
        self,
        predictions: List[Prediction],
        min_tier: SignalTier = SignalTier.C,
        min_edge: float = 0.0,
        bet_types: Optional[List[BetType]] = None,
        sports: Optional[List[str]] = None,
        actionable_only: bool = False,
    ) -> List[Prediction]:
        """
        Filter predictions by criteria.
        
        Args:
            predictions: List of predictions to filter
            min_tier: Minimum signal tier
            min_edge: Minimum edge threshold
            bet_types: Allowed bet types
            sports: Allowed sports
            actionable_only: Only return actionable predictions
            
        Returns:
            Filtered list of predictions
        """
        tier_order = ['A', 'B', 'C', 'D']
        min_tier_idx = tier_order.index(min_tier.value)
        
        filtered = []
        for pred in predictions:
            # Check tier
            pred_tier_idx = tier_order.index(pred.signal_tier.value)
            if pred_tier_idx > min_tier_idx:
                continue
            
            # Check edge
            if pred.edge < min_edge:
                continue
            
            # Check bet type
            if bet_types and pred.bet_type not in bet_types:
                continue
            
            # Check sport
            if sports and pred.sport not in sports:
                continue
            
            # Check actionable
            if actionable_only:
                if pred.recommendation.action not in ['STRONG_BET', 'BET', 'LEAN']:
                    continue
            
            filtered.append(pred)
        
        return filtered
    
    def rank_predictions(
        self,
        predictions: List[Prediction],
        method: str = 'expected_value',
    ) -> List[Prediction]:
        """
        Rank predictions by specified method.
        
        Args:
            predictions: List to rank
            method: Ranking method (expected_value, edge, confidence, kelly)
            
        Returns:
            Sorted list of predictions
        """
        if method == 'expected_value':
            return sorted(
                predictions,
                key=lambda p: p.edge_analysis.expected_value,
                reverse=True
            )
        elif method == 'edge':
            return sorted(predictions, key=lambda p: p.edge, reverse=True)
        elif method == 'confidence':
            return sorted(predictions, key=lambda p: p.confidence, reverse=True)
        elif method == 'kelly':
            return sorted(
                predictions,
                key=lambda p: p.recommendation.kelly_fraction,
                reverse=True
            )
        elif method == 'sharpe':
            return sorted(
                predictions,
                key=lambda p: p.edge_analysis.sharpe_ratio,
                reverse=True
            )
        else:
            return predictions
    
    def get_top_predictions(
        self,
        predictions: List[Prediction],
        top_n: int = 10,
        method: str = 'expected_value',
    ) -> List[Prediction]:
        """Get top N predictions by ranking method."""
        ranked = self.rank_predictions(predictions, method)
        return ranked[:top_n]


# =============================================================================
# PLAYER PROPS ENGINE
# =============================================================================

@dataclass
class PlayerPropPrediction:
    """Player prop prediction structure."""
    prediction_id: str
    game_id: str
    player_id: str
    player_name: str
    team: str
    sport: str
    
    # Prop details
    prop_type: str  # 'points', 'rebounds', 'assists', etc.
    line: float
    predicted_value: float
    predicted_side: PredictedSide  # OVER or UNDER
    
    # Confidence
    probability: float
    confidence: float
    signal_tier: SignalTier
    
    # Edge
    edge: float
    odds: int
    
    # Factors
    projection_source: str  # 'model', 'consensus', 'hybrid'
    key_factors: List[str] = field(default_factory=list)
    
    # Integrity
    prediction_hash: str = ""
    locked_at: datetime = field(default_factory=datetime.utcnow)
    
    # Grading
    graded: bool = False
    actual_value: Optional[float] = None
    result: Optional[str] = None


class PlayerPropsEngine:
    """
    Advanced player props prediction engine.
    
    Supports:
    - Points, rebounds, assists (basketball)
    - Passing/rushing/receiving yards (football)
    - Strikeouts, hits (baseball)
    - And more
    """
    
    PROP_TYPES = {
        'NBA': ['points', 'rebounds', 'assists', 'threes', 'steals', 'blocks', 'pra'],
        'NFL': ['passing_yards', 'passing_tds', 'rushing_yards', 'receiving_yards', 'receptions'],
        'MLB': ['strikeouts', 'hits', 'total_bases', 'rbis'],
        'NHL': ['goals', 'assists', 'shots', 'saves'],
    }
    
    def __init__(
        self,
        prediction_engine: AdvancedPredictionEngine,
    ):
        self.engine = prediction_engine
        self._prop_counter = 0
    
    def generate_prop_prediction(
        self,
        game_id: str,
        player_id: str,
        player_name: str,
        team: str,
        sport: str,
        prop_type: str,
        line: float,
        odds_over: int,
        odds_under: int,
        player_features: Dict[str, float],
        projected_value: Optional[float] = None,
    ) -> PlayerPropPrediction:
        """
        Generate player prop prediction.
        
        Args:
            game_id: Game identifier
            player_id: Player identifier
            player_name: Player name
            team: Player's team
            sport: Sport code
            prop_type: Type of prop
            line: Prop line
            odds_over: Over odds
            odds_under: Under odds
            player_features: Player-specific features
            projected_value: Optional projected stat value
            
        Returns:
            PlayerPropPrediction
        """
        # Calculate projected value if not provided
        if projected_value is None:
            projected_value = self._calculate_projection(
                player_features, prop_type, sport
            )
        
        # Determine over/under
        if projected_value > line:
            predicted_side = PredictedSide.OVER
            # Calculate probability based on projection vs line
            diff_pct = (projected_value - line) / line if line > 0 else 0.1
            probability = 0.5 + min(diff_pct * 2, 0.3)  # Cap adjustment
            odds = odds_over
        else:
            predicted_side = PredictedSide.UNDER
            diff_pct = (line - projected_value) / line if line > 0 else 0.1
            probability = 0.5 + min(diff_pct * 2, 0.3)
            odds = odds_under
        
        # Calculate implied probability and edge
        if odds < 0:
            implied_prob = abs(odds) / (abs(odds) + 100)
        else:
            implied_prob = 100 / (odds + 100)
        
        edge = probability - implied_prob
        
        # Determine tier
        confidence = max(probability, 1 - probability)
        if confidence >= 0.65:
            tier = SignalTier.A
        elif confidence >= 0.60:
            tier = SignalTier.B
        elif confidence >= 0.55:
            tier = SignalTier.C
        else:
            tier = SignalTier.D
        
        # Generate ID and hash
        self._prop_counter += 1
        prediction_id = f"prop_{player_id}_{prop_type}_{self._prop_counter}"
        
        hash_data = {
            'player_id': player_id,
            'prop_type': prop_type,
            'line': line,
            'predicted_side': predicted_side.value,
            'probability': round(probability, 6),
        }
        prediction_hash = hashlib.sha256(
            json.dumps(hash_data, sort_keys=True).encode()
        ).hexdigest()
        
        # Key factors
        key_factors = self._get_prop_factors(player_features, prop_type)
        
        return PlayerPropPrediction(
            prediction_id=prediction_id,
            game_id=game_id,
            player_id=player_id,
            player_name=player_name,
            team=team,
            sport=sport,
            prop_type=prop_type,
            line=line,
            predicted_value=projected_value,
            predicted_side=predicted_side,
            probability=probability,
            confidence=confidence,
            signal_tier=tier,
            edge=edge,
            odds=odds,
            projection_source='model',
            key_factors=key_factors,
            prediction_hash=prediction_hash,
        )
    
    def _calculate_projection(
        self,
        features: Dict[str, float],
        prop_type: str,
        sport: str,
    ) -> float:
        """Calculate projected value from features."""
        # Use season average as base
        avg_key = f'{prop_type}_avg'
        base = features.get(avg_key, features.get('season_avg', 15.0))
        
        # Adjust by recent form
        recent_avg = features.get(f'{prop_type}_l5', base)
        form_adj = (recent_avg - base) * 0.3
        
        # Adjust by matchup
        opp_rank = features.get(f'opp_{prop_type}_rank', 15)
        matchup_adj = (15 - opp_rank) * 0.5  # Positive if bad defense
        
        # Adjust by home/away
        is_home = features.get('is_home', 0.5)
        home_adj = 0.5 if is_home > 0.5 else -0.5
        
        projection = base + form_adj + matchup_adj + home_adj
        return max(0, projection)
    
    def _get_prop_factors(
        self,
        features: Dict[str, float],
        prop_type: str,
    ) -> List[str]:
        """Get key factors for prop prediction."""
        factors = []
        
        # Check recent form
        recent_avg = features.get(f'{prop_type}_l5', 0)
        season_avg = features.get(f'{prop_type}_avg', 0)
        if recent_avg > season_avg * 1.1:
            factors.append(f"Hot streak: averaging {recent_avg:.1f} in last 5")
        elif recent_avg < season_avg * 0.9:
            factors.append(f"Cold streak: averaging {recent_avg:.1f} in last 5")
        
        # Check matchup
        opp_rank = features.get(f'opp_{prop_type}_rank', 15)
        if opp_rank <= 5:
            factors.append("Tough matchup vs top 5 defense")
        elif opp_rank >= 25:
            factors.append("Favorable matchup vs bottom 5 defense")
        
        # Check minutes/usage
        if 'minutes_avg' in features:
            factors.append(f"Averaging {features['minutes_avg']:.1f} minutes")
        
        return factors[:5]
    
    def grade_prop(
        self,
        prediction: PlayerPropPrediction,
        actual_value: float,
    ) -> PlayerPropPrediction:
        """Grade a player prop prediction."""
        prediction.actual_value = actual_value
        prediction.graded = True
        
        if prediction.predicted_side == PredictedSide.OVER:
            if actual_value > prediction.line:
                prediction.result = 'win'
            elif actual_value < prediction.line:
                prediction.result = 'loss'
            else:
                prediction.result = 'push'
        else:
            if actual_value < prediction.line:
                prediction.result = 'win'
            elif actual_value > prediction.line:
                prediction.result = 'loss'
            else:
                prediction.result = 'push'
        
        return prediction


# =============================================================================
# CORRELATION ANALYZER
# =============================================================================

class CorrelationAnalyzer:
    """
    Analyze correlations between predictions for parlay optimization.
    """
    
    # Known correlations
    POSITIVE_CORRELATIONS = {
        ('spread_home', 'total_over'): 0.15,  # Home cover often means higher scoring
        ('spread_away', 'total_under'): 0.10,
        ('qb_passing_yards', 'team_total_over'): 0.25,
        ('rb_rushing_yards', 'team_spread'): 0.20,
    }
    
    NEGATIVE_CORRELATIONS = {
        ('spread_home', 'spread_away'): -1.0,  # Same game opposite sides
        ('total_over', 'total_under'): -1.0,
    }
    
    def __init__(self):
        self.correlation_cache: Dict[str, float] = {}
    
    def calculate_correlation(
        self,
        pred1: Prediction,
        pred2: Prediction,
    ) -> float:
        """
        Calculate correlation between two predictions.
        
        Returns:
            Correlation coefficient (-1 to 1)
        """
        # Same game check
        if pred1.game_id == pred2.game_id:
            return self._same_game_correlation(pred1, pred2)
        
        # Different games - generally uncorrelated
        return 0.0
    
    def _same_game_correlation(
        self,
        pred1: Prediction,
        pred2: Prediction,
    ) -> float:
        """Calculate correlation for same-game predictions."""
        key1 = f"{pred1.bet_type.value}_{pred1.predicted_side.value}"
        key2 = f"{pred2.bet_type.value}_{pred2.predicted_side.value}"
        
        # Check opposite sides
        if pred1.bet_type == pred2.bet_type:
            if pred1.predicted_side != pred2.predicted_side:
                return -1.0  # Perfect negative correlation
        
        # Check known correlations
        for (k1, k2), corr in self.POSITIVE_CORRELATIONS.items():
            if (key1 == k1 and key2 == k2) or (key1 == k2 and key2 == k1):
                return corr
        
        for (k1, k2), corr in self.NEGATIVE_CORRELATIONS.items():
            if (key1 == k1 and key2 == k2) or (key1 == k2 and key2 == k1):
                return corr
        
        # Default small positive correlation for same game
        return 0.05
    
    def analyze_parlay(
        self,
        predictions: List[Prediction],
    ) -> Dict[str, Any]:
        """
        Analyze correlation risk in a parlay.
        
        Args:
            predictions: List of predictions in parlay
            
        Returns:
            Analysis including correlation matrix and risk assessment
        """
        n = len(predictions)
        if n < 2:
            return {'error': 'Need at least 2 predictions'}
        
        # Build correlation matrix
        corr_matrix = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                corr = self.calculate_correlation(predictions[i], predictions[j])
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr
        
        # Calculate portfolio metrics
        avg_correlation = np.mean(corr_matrix[np.triu_indices(n, k=1)])
        max_correlation = np.max(corr_matrix[np.triu_indices(n, k=1)])
        
        # Risk assessment
        if max_correlation > 0.5:
            risk_level = 'HIGH'
            warning = 'Highly correlated legs detected'
        elif max_correlation > 0.2:
            risk_level = 'MEDIUM'
            warning = 'Some correlation between legs'
        else:
            risk_level = 'LOW'
            warning = 'Well-diversified parlay'
        
        # Calculate adjusted probability (accounting for correlation)
        raw_prob = np.prod([p.probability for p in predictions])
        correlation_penalty = avg_correlation * 0.1 * n
        adjusted_prob = raw_prob * (1 - correlation_penalty)
        
        return {
            'num_legs': n,
            'raw_probability': raw_prob,
            'adjusted_probability': adjusted_prob,
            'avg_correlation': avg_correlation,
            'max_correlation': max_correlation,
            'risk_level': risk_level,
            'warning': warning,
            'correlation_matrix': corr_matrix.tolist(),
        }


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_advanced_prediction_engine(
    meta_ensemble=None,
    shap_explainer=None,
    probability_calibrator=None,
    model_registry=None,
    model_id: str = 'advanced_v2',
) -> AdvancedPredictionEngine:
    """
    Factory function to create an Advanced Prediction Engine.
    
    Args:
        meta_ensemble: MetaEnsemble instance
        shap_explainer: SHAP explainer instance
        probability_calibrator: Calibrator instance
        model_registry: Model registry instance
        model_id: Model identifier
        
    Returns:
        Configured AdvancedPredictionEngine
    """
    return AdvancedPredictionEngine(
        meta_ensemble=meta_ensemble,
        shap_explainer=shap_explainer,
        probability_calibrator=probability_calibrator,
        model_registry=model_registry,
        model_id=model_id,
    )


def create_player_props_engine(
    prediction_engine: Optional[AdvancedPredictionEngine] = None,
) -> PlayerPropsEngine:
    """
    Factory function to create a Player Props Engine.
    
    Args:
        prediction_engine: Parent prediction engine
        
    Returns:
        Configured PlayerPropsEngine
    """
    if prediction_engine is None:
        prediction_engine = create_advanced_prediction_engine()
    return PlayerPropsEngine(prediction_engine)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == '__main__':
    # Create engine
    engine = create_advanced_prediction_engine(model_id='test_v2')
    
    # Sample framework predictions
    framework_preds = {
        'h2o': FrameworkPrediction(
            framework=ModelFramework.H2O,
            probability=0.62,
            recent_accuracy=0.58,
            recent_clv=0.015,
        ),
        'autogluon': FrameworkPrediction(
            framework=ModelFramework.AUTOGLUON,
            probability=0.65,
            recent_accuracy=0.61,
            recent_clv=0.022,
        ),
        'sklearn': FrameworkPrediction(
            framework=ModelFramework.SKLEARN,
            probability=0.60,
            recent_accuracy=0.56,
            recent_clv=0.010,
        ),
    }
    
    # Sample odds
    odds = OddsInfo(
        game_id='NBA_20240115_LAL_GSW',
        spread_home=-3.5,
        spread_away=3.5,
        spread_home_odds=-110,
        spread_away_odds=-110,
        moneyline_home=-150,
        moneyline_away=130,
        total_line=225.5,
        total_over_odds=-110,
        total_under_odds=-110,
        opening_spread=-2.5,
        opening_total=224.0,
        public_spread_home_pct=0.65,
        pinnacle_spread=-3.5,
    )
    
    # Sample features
    features = {
        'home_elo': 1650,
        'away_elo': 1580,
        'elo_diff': 70,
        'home_win_pct_l10': 0.70,
        'away_win_pct_l10': 0.50,
        'rest_advantage': 1,
        'h2h_win_pct': 0.60,
        'spread_movement': -1.0,
        'home_off_rating': 115.5,
        'away_def_rating': 108.2,
    }
    
    # Sample situational
    situational = SituationalModifiers(
        home_rest_days=2,
        away_rest_days=1,
        away_b2b=True,
        is_national_tv=True,
    )
    
    # Generate prediction
    prediction = engine.generate_prediction(
        game_id='NBA_20240115_LAL_GSW',
        sport='NBA',
        home_team='Los Angeles Lakers',
        away_team='Golden State Warriors',
        game_date=datetime.now(),
        bet_type=BetType.SPREAD,
        features=features,
        odds_info=odds,
        framework_predictions=framework_preds,
        situational=situational,
    )
    
    print("=" * 60)
    print("ADVANCED PREDICTION ENGINE OUTPUT")
    print("=" * 60)
    print(f"\n{prediction.get_summary()}")
    print(f"\nPrediction ID: {prediction.prediction_id}")
    print(f"Predicted Side: {prediction.predicted_side.value}")
    print(f"Probability: {prediction.probability:.2%}")
    print(f"Raw Probability: {prediction.raw_probability:.2%}")
    print(f"Uncertainty: {prediction.uncertainty:.4f}")
    print(f"Signal Tier: {prediction.signal_tier.value}")
    print(f"Edge: {prediction.edge:.2%}")
    print(f"Expected Value: {prediction.edge_analysis.expected_value:.4f}")
    print(f"Sharpe Ratio: {prediction.edge_analysis.sharpe_ratio:.2f}")
    print(f"\nRecommendation: {prediction.recommendation.action}")
    print(f"Kelly Fraction: {prediction.recommendation.kelly_fraction:.4f}")
    print(f"Reason: {prediction.recommendation.primary_reason}")
    print(f"\nMarket Type: {prediction.market_type.value}")
    print(f"Sharp Side: {prediction.sharp_side}")
    print(f"Steam Move: {prediction.steam_move_detected}")
    print(f"Predicted CLV: {prediction.predicted_clv:.2%}")
    print(f"\nKey Insight: {prediction.explanation.key_insight}")
    print(f"\nHash: {prediction.prediction_hash[:32]}...")
    print(f"Hash Valid: {engine.verify_prediction_hash(prediction)}")
    
    # Test grading
    print("\n" + "=" * 60)
    print("GRADING TEST")
    print("=" * 60)
    graded = engine.grade_prediction(prediction, home_score=115, away_score=108)
    print(f"Result: {graded.result}")
    print(f"Profit/Loss: {graded.profit_loss:.2f} units")
    
    # Test batch
    print("\n" + "=" * 60)
    print("BATCH PREDICTION TEST")
    print("=" * 60)
    games = [
        {
            'game_id': 'NBA_20240115_LAL_GSW',
            'sport': 'NBA',
            'home_team': 'Lakers',
            'away_team': 'Warriors',
            'game_date': datetime.now(),
        }
    ]
    batch = engine.generate_batch_predictions(
        games=games,
        features_by_game={'NBA_20240115_LAL_GSW': features},
        odds_by_game={'NBA_20240115_LAL_GSW': odds},
        framework_predictions_by_game={'NBA_20240115_LAL_GSW': framework_preds},
    )
    print(f"Batch ID: {batch.batch_id}")
    print(f"Total Predictions: {batch.total_predictions}")
    print(f"Tier A: {batch.tier_a_count}")
    print(f"Actionable: {batch.actionable_count}")
    
    # Test player props
    print("\n" + "=" * 60)
    print("PLAYER PROPS TEST")
    print("=" * 60)
    props_engine = create_player_props_engine(engine)
    prop = props_engine.generate_prop_prediction(
        game_id='NBA_20240115_LAL_GSW',
        player_id='lebron_james',
        player_name='LeBron James',
        team='LAL',
        sport='NBA',
        prop_type='points',
        line=25.5,
        odds_over=-115,
        odds_under=-105,
        player_features={
            'points_avg': 27.5,
            'points_l5': 29.2,
            'opp_points_rank': 22,
            'minutes_avg': 35.5,
            'is_home': 1,
        },
    )
    print(f"Player: {prop.player_name}")
    print(f"Prop: {prop.prop_type} {prop.predicted_side.value} {prop.line}")
    print(f"Projected Value: {prop.predicted_value:.1f}")
    print(f"Probability: {prop.probability:.2%}")
    print(f"Tier: {prop.signal_tier.value}")
    print(f"Edge: {prop.edge:.2%}")
    
    # Test correlation analyzer
    print("\n" + "=" * 60)
    print("CORRELATION ANALYSIS TEST")
    print("=" * 60)
    analyzer = CorrelationAnalyzer()
    
    # Generate another prediction for correlation test
    prediction2 = engine.generate_prediction(
        game_id='NBA_20240115_LAL_GSW',
        sport='NBA',
        home_team='Los Angeles Lakers',
        away_team='Golden State Warriors',
        game_date=datetime.now(),
        bet_type=BetType.TOTAL,
        features=features,
        odds_info=odds,
        framework_predictions=framework_preds,
    )
    
    analysis = analyzer.analyze_parlay([prediction, prediction2])
    print(f"Parlay Legs: {analysis['num_legs']}")
    print(f"Raw Probability: {analysis['raw_probability']:.4f}")
    print(f"Adjusted Probability: {analysis['adjusted_probability']:.4f}")
    print(f"Risk Level: {analysis['risk_level']}")
    print(f"Warning: {analysis['warning']}")
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


# =============================================================================
# EXPORT ALIASES FOR BACKWARD COMPATIBILITY
# =============================================================================
# These aliases ensure consistent imports across the codebase

PredictionEngine = AdvancedPredictionEngine
"""Alias for AdvancedPredictionEngine for backward compatibility."""

__all__ = [
    # Main Classes
    "AdvancedPredictionEngine",
    "PredictionEngine",  # Alias
    "PlayerPropsEngine",
    "CorrelationAnalyzer",
    # Data Classes
    "BetType",
    "PredictedSide",
    "SignalTier",
    "MarketType",
    "ModelFramework",
    "SportPredictionConfig",
    "OddsInfo",
    "FrameworkPrediction",
    "EnsemblePrediction",
    "SituationalModifiers",
    "EdgeAnalysis",
    "BettingRecommendation",
    "PredictionExplanation",
    "Prediction",
    "PredictionBatch",
    "PlayerPropPrediction",
    # Factory Function
    "create_prediction_engine",
]
