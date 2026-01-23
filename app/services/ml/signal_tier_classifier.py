"""
ROYALEY - Signal Tier Classifier
Phase 2: Classification of predictions into confidence tiers

Signal tiers help categorize predictions by confidence level:
- Tier A (65%+): Elite predictions, highest edge
- Tier B (60-65%): Strong value plays
- Tier C (55-60%): Moderate confidence
- Tier D (<55%): Lower confidence, track only
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import numpy as np
import pandas as pd

from .config import MLConfig, default_ml_config

logger = logging.getLogger(__name__)


class SignalTier(Enum):
    """Signal tier classification"""
    A = "A"  # Elite - 65%+
    B = "B"  # Strong - 60-65%
    C = "C"  # Moderate - 55-60%
    D = "D"  # Lower - <55%
    
    @property
    def description(self) -> str:
        descriptions = {
            "A": "Elite prediction - Maximum bet sizing",
            "B": "Strong value - Standard bet sizing",
            "C": "Moderate confidence - Reduced bet sizing",
            "D": "Lower confidence - Track only, no betting",
        }
        return descriptions.get(self.value, "Unknown")
    
    @property
    def kelly_multiplier(self) -> float:
        """Kelly criterion multiplier for this tier"""
        multipliers = {
            "A": 1.0,    # Full Kelly fraction
            "B": 0.75,   # 75% of Kelly
            "C": 0.5,    # 50% of Kelly
            "D": 0.0,    # No betting
        }
        return multipliers.get(self.value, 0.0)
    
    @property
    def target_accuracy(self) -> float:
        """Target accuracy for this tier"""
        targets = {
            "A": 0.65,
            "B": 0.60,
            "C": 0.55,
            "D": 0.50,
        }
        return targets.get(self.value, 0.50)


@dataclass
class TierThresholds:
    """Configurable tier thresholds"""
    tier_a_min: float = 0.65
    tier_b_min: float = 0.60
    tier_c_min: float = 0.55
    tier_d_min: float = 0.00  # Everything below C
    
    def to_dict(self) -> Dict:
        return {
            'tier_a_min': self.tier_a_min,
            'tier_b_min': self.tier_b_min,
            'tier_c_min': self.tier_c_min,
            'tier_d_min': self.tier_d_min,
        }


@dataclass
class TierClassification:
    """Result of tier classification for a prediction"""
    tier: SignalTier
    probability: float
    confidence_score: float
    edge: float  # Edge over implied odds
    
    # Betting guidance
    recommended_action: str
    kelly_multiplier: float
    max_bet_pct: float
    
    # Additional info
    is_actionable: bool
    reason: str = ""
    
    # Metadata
    classified_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict:
        return {
            'tier': self.tier.value,
            'probability': self.probability,
            'confidence_score': self.confidence_score,
            'edge': self.edge,
            'recommended_action': self.recommended_action,
            'kelly_multiplier': self.kelly_multiplier,
            'max_bet_pct': self.max_bet_pct,
            'is_actionable': self.is_actionable,
            'reason': self.reason,
        }


@dataclass
class TierPerformanceMetrics:
    """Performance metrics for a specific tier"""
    tier: SignalTier
    total_predictions: int = 0
    correct_predictions: int = 0
    accuracy: float = 0.0
    
    # ROI metrics
    total_wagered: float = 0.0
    total_profit: float = 0.0
    roi: float = 0.0
    
    # CLV metrics
    average_clv: float = 0.0
    positive_clv_pct: float = 0.0
    
    # Time period
    start_date: datetime = None
    end_date: datetime = None
    
    def update(self, outcome: bool, profit: float, clv: float) -> None:
        """Update metrics with new result"""
        self.total_predictions += 1
        if outcome:
            self.correct_predictions += 1
        self.total_wagered += 1.0  # Assume unit bets
        self.total_profit += profit
        
        # Recalculate
        self.accuracy = self.correct_predictions / self.total_predictions
        self.roi = self.total_profit / self.total_wagered if self.total_wagered > 0 else 0
        
        # Update CLV (running average)
        n = self.total_predictions
        self.average_clv = ((n - 1) * self.average_clv + clv) / n
        
        if clv > 0:
            self.positive_clv_pct = (
                (self.positive_clv_pct * (n - 1) + 1) / n
            )


class SignalTierClassifier:
    """
    Classifies predictions into signal tiers based on confidence levels.
    
    Provides:
    - Tier assignment based on probability thresholds
    - Betting recommendations
    - Performance tracking by tier
    """
    
    def __init__(
        self,
        config: MLConfig = None,
        thresholds: TierThresholds = None,
    ):
        """
        Initialize tier classifier.
        
        Args:
            config: ML configuration
            thresholds: Custom tier thresholds
        """
        self.config = config or default_ml_config
        
        # Set thresholds from config or use custom
        if thresholds:
            self.thresholds = thresholds
        else:
            self.thresholds = TierThresholds(
                tier_a_min=getattr(self.config, 'signal_tier_a_min', self.config.tier_a_threshold),
                tier_b_min=getattr(self.config, 'signal_tier_b_min', self.config.tier_b_threshold),
                tier_c_min=getattr(self.config, 'signal_tier_c_min', self.config.tier_c_threshold),
            )
        
        # Performance tracking
        self._tier_metrics: Dict[SignalTier, TierPerformanceMetrics] = {
            tier: TierPerformanceMetrics(tier=tier)
            for tier in SignalTier
        }
        
        # Minimum edge threshold for actionable predictions
        self.min_edge_threshold = getattr(self.config, 'min_edge_threshold', 0.03)
    
    def classify(
        self,
        probability: float,
        implied_probability: float = 0.5,
        calibrated: bool = True,
    ) -> TierClassification:
        """
        Classify a prediction into a signal tier.
        
        Args:
            probability: Predicted probability (0-1)
            implied_probability: Market implied probability
            calibrated: Whether probability has been calibrated
            
        Returns:
            TierClassification with tier and recommendations
        """
        # Ensure probability is in valid range
        probability = max(0.0, min(1.0, probability))
        
        # Determine tier
        if probability >= self.thresholds.tier_a_min:
            tier = SignalTier.A
        elif probability >= self.thresholds.tier_b_min:
            tier = SignalTier.B
        elif probability >= self.thresholds.tier_c_min:
            tier = SignalTier.C
        else:
            tier = SignalTier.D
        
        # Calculate edge
        edge = probability - implied_probability
        
        # Calculate confidence score (how far above tier minimum)
        if tier == SignalTier.A:
            confidence = (probability - self.thresholds.tier_a_min) / (1.0 - self.thresholds.tier_a_min)
        elif tier == SignalTier.B:
            confidence = (probability - self.thresholds.tier_b_min) / (self.thresholds.tier_a_min - self.thresholds.tier_b_min)
        elif tier == SignalTier.C:
            confidence = (probability - self.thresholds.tier_c_min) / (self.thresholds.tier_b_min - self.thresholds.tier_c_min)
        else:
            confidence = probability / self.thresholds.tier_c_min if self.thresholds.tier_c_min > 0 else 0
        
        confidence = max(0.0, min(1.0, confidence))
        
        # Determine if actionable
        is_actionable = (
            tier in [SignalTier.A, SignalTier.B, SignalTier.C] and
            edge >= self.min_edge_threshold
        )
        
        # Determine recommended action
        if tier == SignalTier.A and edge >= self.min_edge_threshold:
            action = "STRONG BET"
            reason = f"Elite prediction with {edge*100:.1f}% edge"
        elif tier == SignalTier.B and edge >= self.min_edge_threshold:
            action = "STANDARD BET"
            reason = f"Strong value with {edge*100:.1f}% edge"
        elif tier == SignalTier.C and edge >= self.min_edge_threshold:
            action = "SMALL BET"
            reason = f"Moderate edge of {edge*100:.1f}%"
        elif edge < self.min_edge_threshold:
            action = "NO BET"
            is_actionable = False
            reason = f"Insufficient edge ({edge*100:.1f}% < {self.min_edge_threshold*100:.1f}% minimum)"
        else:
            action = "TRACK ONLY"
            reason = "Lower confidence prediction"
        
        # Calculate bet sizing limits
        kelly_mult = tier.kelly_multiplier if is_actionable else 0.0
        max_bet = self.config.max_bet_percent * kelly_mult if is_actionable else 0.0
        
        return TierClassification(
            tier=tier,
            probability=probability,
            confidence_score=confidence,
            edge=edge,
            recommended_action=action,
            kelly_multiplier=kelly_mult,
            max_bet_pct=max_bet,
            is_actionable=is_actionable,
            reason=reason,
        )
    
    def classify_batch(
        self,
        probabilities: List[float],
        implied_probabilities: List[float] = None,
    ) -> List[TierClassification]:
        """
        Classify multiple predictions.
        
        Args:
            probabilities: List of predicted probabilities
            implied_probabilities: List of market implied probabilities
            
        Returns:
            List of TierClassification objects
        """
        if implied_probabilities is None:
            implied_probabilities = [0.5] * len(probabilities)
        
        return [
            self.classify(prob, implied)
            for prob, implied in zip(probabilities, implied_probabilities)
        ]
    
    def record_outcome(
        self,
        tier: SignalTier,
        outcome: bool,
        profit: float = 0.0,
        clv: float = 0.0,
    ) -> None:
        """
        Record prediction outcome for performance tracking.
        
        Args:
            tier: Signal tier of the prediction
            outcome: Whether prediction was correct
            profit: Profit/loss from the bet
            clv: Closing line value
        """
        self._tier_metrics[tier].update(outcome, profit, clv)
    
    def get_tier_performance(
        self,
        tier: SignalTier = None,
    ) -> Dict[str, TierPerformanceMetrics]:
        """
        Get performance metrics by tier.
        
        Args:
            tier: Specific tier to get, or None for all
            
        Returns:
            Dictionary of tier -> metrics
        """
        if tier:
            return {tier.value: self._tier_metrics[tier]}
        return {t.value: m for t, m in self._tier_metrics.items()}
    
    def get_tier_distribution(
        self,
        classifications: List[TierClassification],
    ) -> Dict[str, int]:
        """
        Get distribution of predictions across tiers.
        
        Args:
            classifications: List of tier classifications
            
        Returns:
            Dictionary of tier -> count
        """
        distribution = {tier.value: 0 for tier in SignalTier}
        
        for c in classifications:
            distribution[c.tier.value] += 1
        
        return distribution
    
    def filter_by_tier(
        self,
        classifications: List[TierClassification],
        min_tier: SignalTier = None,
        max_tier: SignalTier = None,
        actionable_only: bool = False,
    ) -> List[TierClassification]:
        """
        Filter classifications by tier.
        
        Args:
            classifications: List to filter
            min_tier: Minimum tier (inclusive)
            max_tier: Maximum tier (inclusive)
            actionable_only: Only return actionable predictions
            
        Returns:
            Filtered list
        """
        tier_order = [SignalTier.A, SignalTier.B, SignalTier.C, SignalTier.D]
        
        filtered = []
        for c in classifications:
            # Check actionable
            if actionable_only and not c.is_actionable:
                continue
            
            # Check tier range
            tier_idx = tier_order.index(c.tier)
            
            if min_tier:
                min_idx = tier_order.index(min_tier)
                if tier_idx > min_idx:  # Higher index = lower tier
                    continue
            
            if max_tier:
                max_idx = tier_order.index(max_tier)
                if tier_idx < max_idx:  # Lower index = higher tier
                    continue
            
            filtered.append(c)
        
        return filtered
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics across all tiers"""
        stats = {
            'tier_performance': {},
            'overall': {
                'total_predictions': 0,
                'correct_predictions': 0,
                'accuracy': 0.0,
                'total_profit': 0.0,
                'roi': 0.0,
            }
        }
        
        total_preds = 0
        total_correct = 0
        total_profit = 0.0
        total_wagered = 0.0
        
        for tier, metrics in self._tier_metrics.items():
            stats['tier_performance'][tier.value] = {
                'predictions': metrics.total_predictions,
                'accuracy': metrics.accuracy,
                'roi': metrics.roi,
                'average_clv': metrics.average_clv,
            }
            
            total_preds += metrics.total_predictions
            total_correct += metrics.correct_predictions
            total_profit += metrics.total_profit
            total_wagered += metrics.total_wagered
        
        if total_preds > 0:
            stats['overall']['total_predictions'] = total_preds
            stats['overall']['correct_predictions'] = total_correct
            stats['overall']['accuracy'] = total_correct / total_preds
            stats['overall']['total_profit'] = total_profit
            stats['overall']['roi'] = total_profit / total_wagered if total_wagered > 0 else 0
        
        return stats
    
    def adjust_thresholds(
        self,
        performance_data: pd.DataFrame = None,
        target_tier_a_accuracy: float = 0.65,
        target_tier_b_accuracy: float = 0.60,
    ) -> TierThresholds:
        """
        Dynamically adjust thresholds based on historical performance.
        
        This optimizes thresholds to achieve target accuracy for each tier.
        
        Args:
            performance_data: Historical predictions with outcomes
            target_tier_a_accuracy: Target accuracy for Tier A
            target_tier_b_accuracy: Target accuracy for Tier B
            
        Returns:
            Optimized TierThresholds
        """
        if performance_data is None or len(performance_data) == 0:
            logger.warning("No performance data provided for threshold adjustment")
            return self.thresholds
        
        # Require columns: probability, outcome
        if 'probability' not in performance_data.columns:
            logger.error("performance_data must have 'probability' column")
            return self.thresholds
        if 'outcome' not in performance_data.columns:
            logger.error("performance_data must have 'outcome' column")
            return self.thresholds
        
        # Sort by probability descending
        df = performance_data.sort_values('probability', ascending=False)
        
        # Find threshold for Tier A
        tier_a_threshold = self._find_threshold_for_accuracy(
            df, target_tier_a_accuracy
        )
        
        # Find threshold for Tier B (among remaining)
        df_below_a = df[df['probability'] < tier_a_threshold]
        tier_b_threshold = self._find_threshold_for_accuracy(
            df_below_a, target_tier_b_accuracy
        )
        
        # Tier C threshold remains fixed
        tier_c_threshold = 0.55
        
        # Ensure proper ordering
        tier_a_threshold = max(tier_a_threshold, 0.60)
        tier_b_threshold = max(tier_b_threshold, 0.55)
        tier_b_threshold = min(tier_b_threshold, tier_a_threshold - 0.01)
        
        new_thresholds = TierThresholds(
            tier_a_min=tier_a_threshold,
            tier_b_min=tier_b_threshold,
            tier_c_min=tier_c_threshold,
        )
        
        logger.info(
            f"Adjusted thresholds: A >= {tier_a_threshold:.3f}, "
            f"B >= {tier_b_threshold:.3f}, C >= {tier_c_threshold:.3f}"
        )
        
        return new_thresholds
    
    def _find_threshold_for_accuracy(
        self,
        df: pd.DataFrame,
        target_accuracy: float,
    ) -> float:
        """Find probability threshold that achieves target accuracy"""
        if len(df) < 10:
            return df['probability'].median() if len(df) > 0 else 0.5
        
        # Try different thresholds
        best_threshold = 0.5
        best_diff = float('inf')
        
        for threshold in np.arange(0.50, 0.80, 0.01):
            subset = df[df['probability'] >= threshold]
            if len(subset) < 10:
                continue
            
            accuracy = subset['outcome'].mean()
            diff = abs(accuracy - target_accuracy)
            
            if diff < best_diff:
                best_diff = diff
                best_threshold = threshold
        
        return best_threshold
    
    def reset_metrics(self) -> None:
        """Reset all performance metrics"""
        self._tier_metrics = {
            tier: TierPerformanceMetrics(tier=tier)
            for tier in SignalTier
        }
        logger.info("Reset tier performance metrics")


def assign_signal_tier(probability: float) -> str:
    """
    Simple function to assign signal tier.
    
    Args:
        probability: Predicted probability
        
    Returns:
        Tier letter (A, B, C, or D)
    """
    if probability >= 0.65:
        return 'A'
    elif probability >= 0.60:
        return 'B'
    elif probability >= 0.55:
        return 'C'
    else:
        return 'D'
