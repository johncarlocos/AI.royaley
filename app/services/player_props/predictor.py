"""
ROYALEY - Player Props Prediction Service
Enterprise-grade player performance predictions for props betting.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class PropType(str, Enum):
    """Supported player prop types"""
    # Basketball
    POINTS = "points"
    REBOUNDS = "rebounds"
    ASSISTS = "assists"
    PRA = "points_rebounds_assists"
    THREES = "three_pointers"
    STEALS = "steals"
    BLOCKS = "blocks"
    TURNOVERS = "turnovers"
    
    # Football
    PASSING_YARDS = "passing_yards"
    PASSING_TDS = "passing_touchdowns"
    RUSHING_YARDS = "rushing_yards"
    RECEIVING_YARDS = "receiving_yards"
    RECEPTIONS = "receptions"
    
    # Baseball
    STRIKEOUTS = "strikeouts"
    HITS = "hits"
    TOTAL_BASES = "total_bases"
    RBIS = "rbis"
    RUNS = "runs"
    
    # Hockey
    SHOTS = "shots"
    GOALS = "goals"
    HOCKEY_ASSISTS = "hockey_assists"
    SAVES = "saves"


@dataclass
class PlayerPropPrediction:
    """Player prop prediction with confidence intervals"""
    player_id: str
    player_name: str
    game_id: str
    prop_type: PropType
    line: float
    predicted_value: float
    over_probability: float
    under_probability: float
    confidence: float
    edge: float
    recommended_side: str
    signal_tier: str
    features_used: Dict[str, float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def has_value(self) -> bool:
        """Check if prediction has betting value"""
        return abs(self.edge) >= 0.03  # 3% minimum edge
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "player_id": self.player_id,
            "player_name": self.player_name,
            "game_id": self.game_id,
            "prop_type": self.prop_type.value,
            "line": self.line,
            "predicted_value": round(self.predicted_value, 2),
            "over_probability": round(self.over_probability, 4),
            "under_probability": round(self.under_probability, 4),
            "confidence": round(self.confidence, 4),
            "edge": round(self.edge, 4),
            "recommended_side": self.recommended_side,
            "signal_tier": self.signal_tier,
            "has_value": self.has_value,
            "created_at": self.created_at.isoformat()
        }


class PlayerStatsCalculator:
    """Calculate rolling player statistics"""
    
    def __init__(self, lookback_games: int = 10):
        self.lookback_games = lookback_games
    
    def calculate_rolling_stats(
        self,
        game_logs: List[Dict],
        prop_type: PropType
    ) -> Dict[str, float]:
        """Calculate rolling statistics for a prop type"""
        if not game_logs:
            return {}
        
        # Get values for the prop type
        values = self._extract_prop_values(game_logs, prop_type)
        
        if not values:
            return {}
        
        recent = values[-self.lookback_games:] if len(values) >= self.lookback_games else values
        
        return {
            "season_avg": np.mean(values),
            "season_std": np.std(values),
            "recent_avg": np.mean(recent),
            "recent_std": np.std(recent),
            "recent_median": np.median(recent),
            "games_played": len(values),
            "recent_trend": self._calculate_trend(recent),
            "consistency": 1 - (np.std(recent) / (np.mean(recent) + 0.001)),
            "hit_rate_over_season_avg": np.mean([1 if v > np.mean(values) else 0 for v in recent]),
        }
    
    def _extract_prop_values(
        self,
        game_logs: List[Dict],
        prop_type: PropType
    ) -> List[float]:
        """Extract prop values from game logs"""
        values = []
        
        stat_mapping = {
            PropType.POINTS: "points",
            PropType.REBOUNDS: "rebounds",
            PropType.ASSISTS: "assists",
            PropType.THREES: "three_pointers_made",
            PropType.STEALS: "steals",
            PropType.BLOCKS: "blocks",
            PropType.PASSING_YARDS: "passing_yards",
            PropType.RUSHING_YARDS: "rushing_yards",
            PropType.RECEIVING_YARDS: "receiving_yards",
            PropType.STRIKEOUTS: "strikeouts",
        }
        
        stat_key = stat_mapping.get(prop_type, prop_type.value)
        
        for log in game_logs:
            if stat_key in log:
                values.append(float(log[stat_key]))
            elif prop_type == PropType.PRA:
                pts = log.get("points", 0)
                reb = log.get("rebounds", 0)
                ast = log.get("assists", 0)
                values.append(float(pts + reb + ast))
        
        return values
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend coefficient"""
        if len(values) < 3:
            return 0.0
        
        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)
        return slope


class OpponentAnalyzer:
    """Analyze opponent defensive metrics"""
    
    def get_opponent_defense_factor(
        self,
        opponent_id: str,
        prop_type: PropType,
        league_average: float
    ) -> float:
        """
        Get opponent defense factor
        >1.0 means opponent allows more than average
        <1.0 means opponent is better defensively
        """
        # This would typically query opponent stats from database
        # For now, return neutral factor
        return 1.0
    
    def get_opponent_pace_factor(self, opponent_id: str) -> float:
        """Get opponent pace factor for pace adjustment"""
        return 1.0


class PlayerPropsPredictor:
    """
    Enterprise player props prediction service.
    Uses historical player data, opponent analysis, and situational factors.
    """
    
    def __init__(self):
        self.stats_calculator = PlayerStatsCalculator()
        self.opponent_analyzer = OpponentAnalyzer()
        self.min_games_required = 5
    
    def predict_prop(
        self,
        player_id: str,
        player_name: str,
        game_id: str,
        prop_type: PropType,
        line: float,
        game_logs: List[Dict],
        opponent_id: Optional[str] = None,
        is_home: bool = True,
        rest_days: int = 2
    ) -> PlayerPropPrediction:
        """Generate prediction for a player prop"""
        
        # Calculate player statistics
        stats = self.stats_calculator.calculate_rolling_stats(game_logs, prop_type)
        
        if not stats or stats.get("games_played", 0) < self.min_games_required:
            # Not enough data for reliable prediction
            return self._create_low_confidence_prediction(
                player_id, player_name, game_id, prop_type, line
            )
        
        # Base prediction from recent average
        base_prediction = stats["recent_avg"]
        
        # Apply adjustments
        adjustments = []
        
        # Home/Away adjustment
        home_adj = 1.02 if is_home else 0.98
        adjustments.append(("home_away", home_adj))
        
        # Rest adjustment
        if rest_days == 0:  # Back-to-back
            rest_adj = 0.95
        elif rest_days >= 3:  # Well rested
            rest_adj = 1.02
        else:
            rest_adj = 1.0
        adjustments.append(("rest", rest_adj))
        
        # Trend adjustment
        trend = stats.get("recent_trend", 0)
        trend_adj = 1 + (trend / (stats["recent_avg"] + 0.001)) * 0.1
        trend_adj = max(0.95, min(1.05, trend_adj))  # Cap adjustment
        adjustments.append(("trend", trend_adj))
        
        # Opponent adjustment
        if opponent_id:
            opp_factor = self.opponent_analyzer.get_opponent_defense_factor(
                opponent_id, prop_type, stats["season_avg"]
            )
            adjustments.append(("opponent", opp_factor))
        
        # Apply all adjustments
        predicted_value = base_prediction
        for name, adj in adjustments:
            predicted_value *= adj
        
        # Calculate probabilities using normal distribution
        std = stats["recent_std"]
        if std == 0:
            std = stats["season_std"]
        if std == 0:
            std = predicted_value * 0.2  # Fallback: 20% of prediction
        
        # Over probability = P(X > line)
        z_score = (line - predicted_value) / std
        over_prob = 1 - self._normal_cdf(z_score)
        under_prob = 1 - over_prob
        
        # Calculate edge
        implied_over = 0.5  # Assuming -110/-110 odds
        if over_prob > under_prob:
            edge = over_prob - implied_over
            recommended_side = "over"
        else:
            edge = under_prob - implied_over
            recommended_side = "under"
        
        # Calculate confidence based on sample size and consistency
        sample_factor = min(1.0, stats["games_played"] / 20)
        consistency_factor = stats.get("consistency", 0.5)
        confidence = 0.5 + (sample_factor * 0.25) + (consistency_factor * 0.25)
        
        # Assign signal tier
        signal_tier = self._assign_tier(over_prob if recommended_side == "over" else under_prob)
        
        # Feature dictionary
        features = {
            "season_avg": stats["season_avg"],
            "recent_avg": stats["recent_avg"],
            "recent_std": stats["recent_std"],
            "trend": stats.get("recent_trend", 0),
            "is_home": float(is_home),
            "rest_days": rest_days,
        }
        for name, adj in adjustments:
            features[f"adj_{name}"] = adj
        
        return PlayerPropPrediction(
            player_id=player_id,
            player_name=player_name,
            game_id=game_id,
            prop_type=prop_type,
            line=line,
            predicted_value=predicted_value,
            over_probability=over_prob,
            under_probability=under_prob,
            confidence=confidence,
            edge=edge,
            recommended_side=recommended_side,
            signal_tier=signal_tier,
            features_used=features
        )
    
    def predict_batch(
        self,
        requests: List[Dict[str, Any]]
    ) -> List[PlayerPropPrediction]:
        """Generate predictions for multiple props"""
        predictions = []
        
        for req in requests:
            try:
                pred = self.predict_prop(
                    player_id=req["player_id"],
                    player_name=req["player_name"],
                    game_id=req["game_id"],
                    prop_type=PropType(req["prop_type"]),
                    line=req["line"],
                    game_logs=req.get("game_logs", []),
                    opponent_id=req.get("opponent_id"),
                    is_home=req.get("is_home", True),
                    rest_days=req.get("rest_days", 2)
                )
                predictions.append(pred)
            except Exception as e:
                logger.error(f"Error predicting prop for {req.get('player_name')}: {e}")
        
        return predictions
    
    def _create_low_confidence_prediction(
        self,
        player_id: str,
        player_name: str,
        game_id: str,
        prop_type: PropType,
        line: float
    ) -> PlayerPropPrediction:
        """Create a low-confidence prediction when data is insufficient"""
        return PlayerPropPrediction(
            player_id=player_id,
            player_name=player_name,
            game_id=game_id,
            prop_type=prop_type,
            line=line,
            predicted_value=line,  # Predict the line itself
            over_probability=0.5,
            under_probability=0.5,
            confidence=0.3,
            edge=0.0,
            recommended_side="none",
            signal_tier="D"
        )
    
    def _normal_cdf(self, z: float) -> float:
        """Cumulative distribution function for standard normal"""
        import math
        return 0.5 * (1 + math.erf(z / math.sqrt(2)))
    
    def _assign_tier(self, probability: float) -> str:
        """Assign signal tier based on probability"""
        if probability >= 0.65:
            return "A"
        elif probability >= 0.60:
            return "B"
        elif probability >= 0.55:
            return "C"
        return "D"


# Global instance
player_props_predictor = PlayerPropsPredictor()
