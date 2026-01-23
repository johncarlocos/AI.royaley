"""
ROYALEY - Advanced Kelly Criterion Calculator
Phase 2: Enterprise-Grade Bet Sizing System

Implements Kelly Criterion with multiple strategies:
- Full Kelly
- Fractional Kelly (1/4, 1/2, 3/4)
- Dynamic Kelly based on confidence
- Simultaneous Kelly for correlated bets
- Risk-adjusted Kelly with drawdown limits
"""

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class KellyStrategy(Enum):
    """Kelly criterion strategy variants."""
    FULL = 'full'           # Full Kelly (aggressive)
    THREE_QUARTER = 'three_quarter'  # 75% Kelly
    HALF = 'half'           # 50% Kelly (conservative)
    QUARTER = 'quarter'     # 25% Kelly (very conservative)
    DYNAMIC = 'dynamic'     # Adjusted by confidence/tier
    FIXED = 'fixed'         # Fixed percentage regardless of edge


class RiskProfile(Enum):
    """User risk tolerance profiles."""
    AGGRESSIVE = 'aggressive'    # Higher variance, higher potential
    MODERATE = 'moderate'        # Balanced approach
    CONSERVATIVE = 'conservative'  # Lower variance, steadier growth
    
    @property
    def kelly_multiplier(self) -> float:
        """Kelly multiplier for this risk profile."""
        return {
            'aggressive': 0.50,
            'moderate': 0.25,
            'conservative': 0.125,
        }.get(self.value, 0.25)
    
    @property
    def max_single_bet(self) -> float:
        """Maximum single bet as % of bankroll."""
        return {
            'aggressive': 0.05,
            'moderate': 0.02,
            'conservative': 0.01,
        }.get(self.value, 0.02)
    
    @property
    def max_daily_exposure(self) -> float:
        """Maximum daily exposure as % of bankroll."""
        return {
            'aggressive': 0.20,
            'moderate': 0.10,
            'conservative': 0.05,
        }.get(self.value, 0.10)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class BetSizing:
    """Complete bet sizing recommendation."""
    # Core sizing
    full_kelly_fraction: float
    recommended_fraction: float
    recommended_units: float
    recommended_amount: float
    
    # Limits applied
    max_bet_applied: bool = False
    daily_limit_applied: bool = False
    
    # Risk metrics
    expected_value: float = 0.0
    variance: float = 0.0
    sharpe_ratio: float = 0.0
    
    # Kelly details
    strategy_used: KellyStrategy = KellyStrategy.QUARTER
    edge: float = 0.0
    probability: float = 0.0
    decimal_odds: float = 0.0
    
    # Warnings
    warnings: List[str] = field(default_factory=list)
    
    # Confidence
    sizing_confidence: float = 0.0  # How confident in sizing (0-1)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'full_kelly': round(self.full_kelly_fraction, 4),
            'recommended_fraction': round(self.recommended_fraction, 4),
            'recommended_units': round(self.recommended_units, 2),
            'recommended_amount': round(self.recommended_amount, 2),
            'expected_value': round(self.expected_value, 4),
            'strategy': self.strategy_used.value,
            'edge': round(self.edge, 4),
            'warnings': self.warnings,
        }


@dataclass
class BankrollState:
    """Current bankroll state."""
    total_bankroll: float
    available_bankroll: float  # After pending bets
    
    # Today's activity
    daily_wagered: float = 0.0
    daily_exposure: float = 0.0
    daily_won: float = 0.0
    daily_lost: float = 0.0
    
    # Pending bets
    pending_bets_count: int = 0
    pending_exposure: float = 0.0
    
    # Historical
    peak_bankroll: float = 0.0
    low_bankroll: float = 0.0
    current_drawdown: float = 0.0
    
    # Risk limits
    daily_limit_remaining: float = 0.0
    
    def update_drawdown(self):
        """Update current drawdown calculation."""
        if self.peak_bankroll > 0:
            self.current_drawdown = (
                (self.peak_bankroll - self.total_bankroll) / self.peak_bankroll
            )
        else:
            self.current_drawdown = 0.0


@dataclass
class SimultaneousBetAnalysis:
    """Analysis for multiple simultaneous bets."""
    total_exposure: float
    individual_sizings: List[BetSizing]
    
    # Correlation adjustment
    correlation_adjustment: float = 0.0
    adjusted_total: float = 0.0
    
    # Risk metrics
    combined_ev: float = 0.0
    portfolio_variance: float = 0.0
    
    # Warnings
    warnings: List[str] = field(default_factory=list)


# =============================================================================
# KELLY CRITERION CALCULATOR
# =============================================================================

class KellyCriterionCalculator:
    """
    Advanced Kelly Criterion calculator with multiple strategies.
    
    Features:
    - Full and fractional Kelly
    - Dynamic Kelly based on confidence/tier
    - Simultaneous bet adjustment
    - Drawdown-based risk adjustment
    - Correlation handling for parlays
    - Sport-specific configurations
    """
    
    # Default configuration
    DEFAULT_KELLY_FRACTION = 0.25
    DEFAULT_MAX_BET = 0.02
    DEFAULT_MIN_EDGE = 0.02
    DEFAULT_MIN_PROBABILITY = 0.52
    
    def __init__(
        self,
        default_strategy: KellyStrategy = KellyStrategy.QUARTER,
        risk_profile: RiskProfile = RiskProfile.MODERATE,
        max_single_bet: float = 0.02,
        min_edge_threshold: float = 0.02,
        enable_drawdown_adjustment: bool = True,
        drawdown_reduction_threshold: float = 0.10,
    ):
        """
        Initialize Kelly calculator.
        
        Args:
            default_strategy: Default Kelly strategy
            risk_profile: User risk profile
            max_single_bet: Maximum single bet as fraction of bankroll
            min_edge_threshold: Minimum edge required to bet
            enable_drawdown_adjustment: Reduce sizing during drawdowns
            drawdown_reduction_threshold: Drawdown % to trigger reduction
        """
        self.default_strategy = default_strategy
        self.risk_profile = risk_profile
        self.max_single_bet = max_single_bet
        self.min_edge_threshold = min_edge_threshold
        self.enable_drawdown_adjustment = enable_drawdown_adjustment
        self.drawdown_reduction_threshold = drawdown_reduction_threshold
        
        # Sport-specific configs
        self._sport_configs: Dict[str, Dict] = {
            'NFL': {'max_bet': 0.025, 'min_edge': 0.03},
            'NBA': {'max_bet': 0.02, 'min_edge': 0.02},
            'MLB': {'max_bet': 0.02, 'min_edge': 0.025},
            'NHL': {'max_bet': 0.02, 'min_edge': 0.025},
            'NCAAF': {'max_bet': 0.015, 'min_edge': 0.025},
            'NCAAB': {'max_bet': 0.015, 'min_edge': 0.025},
        }
        
        logger.info(
            f"Kelly Calculator initialized: {default_strategy.value} strategy, "
            f"{risk_profile.value} profile"
        )
    
    def calculate_bet_size(
        self,
        probability: float,
        american_odds: int,
        bankroll: float,
        signal_tier: str = 'C',
        sport: Optional[str] = None,
        strategy: Optional[KellyStrategy] = None,
        bankroll_state: Optional[BankrollState] = None,
    ) -> BetSizing:
        """
        Calculate optimal bet size using Kelly Criterion.
        
        Args:
            probability: Model probability of winning
            american_odds: American odds for the bet
            bankroll: Current bankroll
            signal_tier: Prediction signal tier (A/B/C/D)
            sport: Sport code for sport-specific limits
            strategy: Override default strategy
            bankroll_state: Current bankroll state for advanced limits
            
        Returns:
            BetSizing recommendation
        """
        # Convert odds
        decimal_odds = self._american_to_decimal(american_odds)
        implied_prob = self._american_to_implied(american_odds)
        
        # Calculate edge
        edge = probability - implied_prob
        
        # Use provided strategy or default
        strategy = strategy or self.default_strategy
        
        # Get sport-specific config
        sport_config = self._sport_configs.get(sport, {})
        max_bet = sport_config.get('max_bet', self.max_single_bet)
        min_edge = sport_config.get('min_edge', self.min_edge_threshold)
        
        warnings = []
        
        # Check minimum edge
        if edge < min_edge:
            warnings.append(f"Edge {edge:.2%} below minimum {min_edge:.2%}")
            return BetSizing(
                full_kelly_fraction=0.0,
                recommended_fraction=0.0,
                recommended_units=0.0,
                recommended_amount=0.0,
                edge=edge,
                probability=probability,
                decimal_odds=decimal_odds,
                strategy_used=strategy,
                warnings=warnings,
                sizing_confidence=0.0,
            )
        
        # Check minimum probability
        if probability < self.DEFAULT_MIN_PROBABILITY:
            warnings.append(f"Probability {probability:.2%} below minimum threshold")
        
        # Calculate full Kelly
        # Kelly formula: f* = (bp - q) / b
        # where b = decimal odds - 1, p = win prob, q = 1 - p
        b = decimal_odds - 1
        q = 1 - probability
        
        if b > 0:
            full_kelly = (b * probability - q) / b
        else:
            full_kelly = 0.0
        
        # Apply strategy multiplier
        kelly_multiplier = self._get_strategy_multiplier(strategy, signal_tier)
        recommended_fraction = full_kelly * kelly_multiplier
        
        # Apply tier-specific adjustment
        tier_multiplier = self._get_tier_multiplier(signal_tier)
        recommended_fraction *= tier_multiplier
        
        # Apply drawdown adjustment if enabled
        drawdown_multiplier = 1.0
        if self.enable_drawdown_adjustment and bankroll_state:
            drawdown_multiplier = self._get_drawdown_multiplier(bankroll_state)
            if drawdown_multiplier < 1.0:
                warnings.append(f"Sizing reduced by {(1-drawdown_multiplier):.0%} due to drawdown")
        
        recommended_fraction *= drawdown_multiplier
        
        # Apply maximum bet limit
        max_bet_applied = False
        if recommended_fraction > max_bet:
            recommended_fraction = max_bet
            max_bet_applied = True
            warnings.append(f"Capped at maximum bet {max_bet:.2%}")
        
        # Check daily limits
        daily_limit_applied = False
        if bankroll_state and bankroll_state.daily_limit_remaining > 0:
            max_for_today = bankroll_state.daily_limit_remaining / bankroll
            if recommended_fraction > max_for_today:
                recommended_fraction = max_for_today
                daily_limit_applied = True
                warnings.append("Limited by daily exposure cap")
        
        # Calculate recommended amount
        recommended_amount = bankroll * recommended_fraction
        recommended_units = recommended_fraction * 100  # As percentage points
        
        # Calculate risk metrics
        expected_value = edge * recommended_amount
        variance = recommended_amount ** 2 * probability * (1 - probability)
        sharpe_ratio = edge / np.sqrt(probability * (1 - probability)) if probability > 0 else 0
        
        # Calculate sizing confidence
        sizing_confidence = self._calculate_sizing_confidence(
            edge, probability, full_kelly, signal_tier
        )
        
        return BetSizing(
            full_kelly_fraction=full_kelly,
            recommended_fraction=recommended_fraction,
            recommended_units=recommended_units,
            recommended_amount=recommended_amount,
            max_bet_applied=max_bet_applied,
            daily_limit_applied=daily_limit_applied,
            expected_value=expected_value,
            variance=variance,
            sharpe_ratio=sharpe_ratio,
            strategy_used=strategy,
            edge=edge,
            probability=probability,
            decimal_odds=decimal_odds,
            warnings=warnings,
            sizing_confidence=sizing_confidence,
        )
    
    def calculate_simultaneous_bets(
        self,
        bets: List[Dict[str, Any]],
        bankroll: float,
        correlation_matrix: Optional[np.ndarray] = None,
    ) -> SimultaneousBetAnalysis:
        """
        Calculate sizing for multiple simultaneous bets.
        
        Args:
            bets: List of bet dictionaries with probability, odds, etc.
            bankroll: Total bankroll
            correlation_matrix: Correlation between bets (optional)
            
        Returns:
            SimultaneousBetAnalysis with adjusted sizings
        """
        individual_sizings = []
        total_exposure = 0.0
        combined_ev = 0.0
        warnings = []
        
        # Calculate individual sizings
        for bet in bets:
            sizing = self.calculate_bet_size(
                probability=bet['probability'],
                american_odds=bet['odds'],
                bankroll=bankroll,
                signal_tier=bet.get('tier', 'C'),
                sport=bet.get('sport'),
            )
            individual_sizings.append(sizing)
            total_exposure += sizing.recommended_fraction
            combined_ev += sizing.expected_value
        
        # Check total exposure
        max_total = self.risk_profile.max_daily_exposure
        
        correlation_adjustment = 0.0
        adjusted_total = total_exposure
        
        if total_exposure > max_total:
            # Scale down proportionally
            scale_factor = max_total / total_exposure
            for sizing in individual_sizings:
                sizing.recommended_fraction *= scale_factor
                sizing.recommended_amount *= scale_factor
                sizing.recommended_units *= scale_factor
            
            adjusted_total = max_total
            warnings.append(f"Total exposure scaled from {total_exposure:.2%} to {max_total:.2%}")
        
        # Apply correlation adjustment if provided
        if correlation_matrix is not None and len(bets) > 1:
            # Reduce sizing for correlated bets
            avg_correlation = np.mean(np.abs(correlation_matrix[np.triu_indices(len(bets), 1)]))
            if avg_correlation > 0.2:
                correlation_factor = 1.0 - (avg_correlation * 0.5)
                for sizing in individual_sizings:
                    sizing.recommended_fraction *= correlation_factor
                    sizing.recommended_amount *= correlation_factor
                
                correlation_adjustment = 1.0 - correlation_factor
                adjusted_total *= correlation_factor
                warnings.append(f"Correlated bets reduced by {correlation_adjustment:.0%}")
        
        # Calculate portfolio variance
        portfolio_variance = sum(s.variance for s in individual_sizings)
        if correlation_matrix is not None:
            # Add correlation terms
            for i in range(len(individual_sizings)):
                for j in range(i + 1, len(individual_sizings)):
                    cov = (
                        individual_sizings[i].recommended_amount *
                        individual_sizings[j].recommended_amount *
                        correlation_matrix[i, j]
                    )
                    portfolio_variance += 2 * cov
        
        return SimultaneousBetAnalysis(
            total_exposure=total_exposure,
            individual_sizings=individual_sizings,
            correlation_adjustment=correlation_adjustment,
            adjusted_total=adjusted_total,
            combined_ev=combined_ev,
            portfolio_variance=portfolio_variance,
            warnings=warnings,
        )
    
    def calculate_parlay_size(
        self,
        legs: List[Dict[str, Any]],
        parlay_odds: int,
        bankroll: float,
    ) -> BetSizing:
        """
        Calculate bet size for a parlay.
        
        Args:
            legs: List of parlay legs with probabilities
            parlay_odds: Combined parlay odds
            bankroll: Current bankroll
            
        Returns:
            BetSizing for the parlay
        """
        # Calculate combined probability (assuming independence)
        combined_prob = np.prod([leg['probability'] for leg in legs])
        
        # Apply correlation discount (parlays tend to be correlated)
        correlation_discount = 0.95 ** (len(legs) - 1)  # 5% discount per additional leg
        adjusted_prob = combined_prob * correlation_discount
        
        # Parlays should use more conservative sizing
        sizing = self.calculate_bet_size(
            probability=adjusted_prob,
            american_odds=parlay_odds,
            bankroll=bankroll,
            signal_tier='C',  # Always treat parlays as lower confidence
            strategy=KellyStrategy.QUARTER,  # Most conservative
        )
        
        # Additional parlay warnings
        if len(legs) > 3:
            sizing.warnings.append("4+ leg parlays are high variance")
        
        return sizing
    
    def _get_strategy_multiplier(
        self,
        strategy: KellyStrategy,
        tier: str,
    ) -> float:
        """Get Kelly multiplier for strategy."""
        base_multipliers = {
            KellyStrategy.FULL: 1.0,
            KellyStrategy.THREE_QUARTER: 0.75,
            KellyStrategy.HALF: 0.50,
            KellyStrategy.QUARTER: 0.25,
            KellyStrategy.FIXED: 0.25,
        }
        
        if strategy == KellyStrategy.DYNAMIC:
            # Adjust based on tier
            tier_adjustments = {
                'A': 0.40,  # More aggressive for Tier A
                'B': 0.30,
                'C': 0.20,
                'D': 0.10,
            }
            return tier_adjustments.get(tier, 0.25)
        
        return base_multipliers.get(strategy, 0.25)
    
    def _get_tier_multiplier(self, tier: str) -> float:
        """Get additional multiplier based on signal tier."""
        multipliers = {
            'A': 1.0,    # Full sizing
            'B': 0.85,   # 85% of recommended
            'C': 0.70,   # 70% of recommended
            'D': 0.0,    # No betting
        }
        return multipliers.get(tier, 0.5)
    
    def _get_drawdown_multiplier(self, state: BankrollState) -> float:
        """Calculate sizing reduction based on drawdown."""
        state.update_drawdown()
        
        if state.current_drawdown < self.drawdown_reduction_threshold:
            return 1.0
        
        # Linear reduction from 1.0 at threshold to 0.5 at 2x threshold
        excess = state.current_drawdown - self.drawdown_reduction_threshold
        reduction = min(0.5, excess / self.drawdown_reduction_threshold)
        
        return max(0.5, 1.0 - reduction)
    
    def _calculate_sizing_confidence(
        self,
        edge: float,
        probability: float,
        full_kelly: float,
        tier: str,
    ) -> float:
        """Calculate confidence in the sizing recommendation."""
        # Higher confidence for higher edge and tier
        edge_score = min(1.0, edge / 0.10)  # 10% edge = max score
        tier_scores = {'A': 1.0, 'B': 0.8, 'C': 0.6, 'D': 0.3}
        tier_score = tier_scores.get(tier, 0.5)
        
        # Penalize extreme Kelly recommendations
        kelly_score = 1.0 if 0 < full_kelly < 0.10 else 0.7
        
        return (edge_score * 0.4 + tier_score * 0.4 + kelly_score * 0.2)
    
    def _american_to_decimal(self, american: int) -> float:
        """Convert American odds to decimal."""
        if american < 0:
            return 1 + (100 / abs(american))
        else:
            return 1 + (american / 100)
    
    def _american_to_implied(self, american: int) -> float:
        """Convert American odds to implied probability."""
        if american < 0:
            return abs(american) / (abs(american) + 100)
        else:
            return 100 / (american + 100)


# =============================================================================
# BANKROLL MANAGER
# =============================================================================

class BankrollManager:
    """
    Manages bankroll state and tracks betting activity.
    
    Features:
    - Real-time bankroll tracking
    - Daily limits management
    - Drawdown monitoring
    - Win/loss streaks
    """
    
    def __init__(
        self,
        initial_bankroll: float,
        daily_limit_percent: float = 0.10,
    ):
        """
        Initialize bankroll manager.
        
        Args:
            initial_bankroll: Starting bankroll
            daily_limit_percent: Maximum daily wagering as % of bankroll
        """
        self.initial_bankroll = initial_bankroll
        self.daily_limit_percent = daily_limit_percent
        
        # Current state
        self.state = BankrollState(
            total_bankroll=initial_bankroll,
            available_bankroll=initial_bankroll,
            peak_bankroll=initial_bankroll,
            low_bankroll=initial_bankroll,
            daily_limit_remaining=initial_bankroll * daily_limit_percent,
        )
        
        # Transaction history
        self._transactions: List[Dict] = []
        self._daily_reset_date: datetime = datetime.utcnow().date()
    
    def place_bet(
        self,
        amount: float,
        prediction_id: str,
    ) -> bool:
        """
        Record a bet placement.
        
        Returns True if bet was accepted, False if limits exceeded.
        """
        self._check_daily_reset()
        
        if amount > self.state.available_bankroll:
            return False
        
        if amount > self.state.daily_limit_remaining:
            return False
        
        # Record transaction
        self._transactions.append({
            'type': 'bet',
            'amount': -amount,
            'prediction_id': prediction_id,
            'timestamp': datetime.utcnow(),
        })
        
        # Update state
        self.state.available_bankroll -= amount
        self.state.daily_wagered += amount
        self.state.daily_exposure += amount
        self.state.daily_limit_remaining -= amount
        self.state.pending_bets_count += 1
        self.state.pending_exposure += amount
        
        return True
    
    def settle_bet(
        self,
        prediction_id: str,
        profit_loss: float,
        stake: float,
    ):
        """
        Settle a bet with result.
        
        Args:
            prediction_id: Prediction identifier
            profit_loss: Net profit (positive) or loss (negative)
            stake: Original stake amount
        """
        # Record transaction
        self._transactions.append({
            'type': 'settlement',
            'amount': stake + profit_loss,
            'prediction_id': prediction_id,
            'profit_loss': profit_loss,
            'timestamp': datetime.utcnow(),
        })
        
        # Update state
        self.state.total_bankroll += profit_loss
        self.state.available_bankroll += stake + profit_loss
        self.state.pending_bets_count -= 1
        self.state.pending_exposure -= stake
        
        if profit_loss > 0:
            self.state.daily_won += profit_loss
        else:
            self.state.daily_lost += abs(profit_loss)
        
        # Update peak/low
        if self.state.total_bankroll > self.state.peak_bankroll:
            self.state.peak_bankroll = self.state.total_bankroll
        if self.state.total_bankroll < self.state.low_bankroll:
            self.state.low_bankroll = self.state.total_bankroll
        
        self.state.update_drawdown()
    
    def deposit(self, amount: float):
        """Add funds to bankroll."""
        self._transactions.append({
            'type': 'deposit',
            'amount': amount,
            'timestamp': datetime.utcnow(),
        })
        
        self.state.total_bankroll += amount
        self.state.available_bankroll += amount
    
    def withdraw(self, amount: float) -> bool:
        """Withdraw funds from bankroll."""
        if amount > self.state.available_bankroll:
            return False
        
        self._transactions.append({
            'type': 'withdrawal',
            'amount': -amount,
            'timestamp': datetime.utcnow(),
        })
        
        self.state.total_bankroll -= amount
        self.state.available_bankroll -= amount
        
        return True
    
    def get_roi(self) -> float:
        """Calculate return on investment."""
        if self.initial_bankroll > 0:
            return (self.state.total_bankroll - self.initial_bankroll) / self.initial_bankroll
        return 0.0
    
    def _check_daily_reset(self):
        """Reset daily limits if new day."""
        today = datetime.utcnow().date()
        if today > self._daily_reset_date:
            self.state.daily_wagered = 0.0
            self.state.daily_exposure = 0.0
            self.state.daily_won = 0.0
            self.state.daily_lost = 0.0
            self.state.daily_limit_remaining = (
                self.state.total_bankroll * self.daily_limit_percent
            )
            self._daily_reset_date = today


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == '__main__':
    # Initialize
    bankroll = BankrollManager(initial_bankroll=10000)
    kelly = KellyCriterionCalculator(
        default_strategy=KellyStrategy.QUARTER,
        risk_profile=RiskProfile.MODERATE,
    )
    
    # Calculate bet size
    sizing = kelly.calculate_bet_size(
        probability=0.58,
        american_odds=-110,
        bankroll=bankroll.state.total_bankroll,
        signal_tier='B',
        sport='NBA',
    )
    
    print("=== Single Bet Sizing ===")
    print(f"Full Kelly: {sizing.full_kelly_fraction:.2%}")
    print(f"Recommended: {sizing.recommended_fraction:.2%}")
    print(f"Amount: ${sizing.recommended_amount:.2f}")
    print(f"Expected Value: ${sizing.expected_value:.2f}")
    print(f"Warnings: {sizing.warnings}")
    
    # Place bet
    if bankroll.place_bet(sizing.recommended_amount, 'pred_001'):
        print(f"\nBet placed: ${sizing.recommended_amount:.2f}")
        print(f"Available: ${bankroll.state.available_bankroll:.2f}")
    
    # Settle bet (win)
    profit = sizing.recommended_amount * 0.91  # -110 odds
    bankroll.settle_bet('pred_001', profit, sizing.recommended_amount)
    
    print(f"\nBet won! Profit: ${profit:.2f}")
    print(f"New bankroll: ${bankroll.state.total_bankroll:.2f}")
    print(f"ROI: {bankroll.get_roi():.2%}")
    
    # Calculate for multiple simultaneous bets
    print("\n=== Multiple Bets ===")
    bets = [
        {'probability': 0.58, 'odds': -110, 'tier': 'B', 'sport': 'NBA'},
        {'probability': 0.55, 'odds': -105, 'tier': 'C', 'sport': 'NBA'},
        {'probability': 0.62, 'odds': -120, 'tier': 'A', 'sport': 'NFL'},
    ]
    
    analysis = kelly.calculate_simultaneous_bets(
        bets=bets,
        bankroll=bankroll.state.total_bankroll,
    )
    
    print(f"Total exposure: {analysis.total_exposure:.2%}")
    print(f"Adjusted total: {analysis.adjusted_total:.2%}")
    print(f"Combined EV: ${analysis.combined_ev:.2f}")
    print(f"Warnings: {analysis.warnings}")


# =============================================================================
# EXPORT ALIASES FOR BACKWARD COMPATIBILITY
# =============================================================================

KellyCalculator = KellyCriterionCalculator
"""Alias for KellyCriterionCalculator for backward compatibility."""

__all__ = [
    # Main Classes
    "KellyCriterionCalculator",
    "KellyCalculator",  # Alias
    "BankrollManager",
    # Data Classes
    "KellyStrategy",
    "RiskProfile",
    "BetSizing",
    "BankrollState",
    "SimultaneousBetAnalysis",
]
