"""
ROYALEY - Betting Services Module
Phase 2: Enterprise-Grade Betting System

This module provides comprehensive betting services including:
- CLV (Closing Line Value) calculation and tracking
- Kelly Criterion bet sizing
- Automatic prediction grading
- Line movement analysis and sharp money detection
"""

__version__ = '2.0.0'

# CLV Calculator
from .clv_calculator import (
    CLVCalculator,
    ClosingLineTracker,
    CLVTier,
    BookmakerSharpness,
    ClosingLine,
    CLVResult,
    CLVPerformance,
)

# Kelly Criterion Calculator
from .kelly_calculator import (
    KellyCriterionCalculator,
    BankrollManager,
    KellyStrategy,
    RiskProfile,
    BetSizing,
    BankrollState,
    SimultaneousBetAnalysis,
)

# Auto-Grader
from .auto_grader import (
    AutoGrader,
    GradeResult,
    BetType,
    GameResult,
    GradedPrediction,
    GradingReport,
    PerformanceMetrics,
)

# Line Movement Analyzer
from .line_movement_analyzer import (
    LineMovementAnalyzer,
    LineMovementType,
    MarketSentiment,
    AlertSeverity,
    OddsSnapshot,
    LineMovement,
    SharpIndicator,
    MarketAlert,
    MarketAnalysis,
)

__all__ = [
    # Version
    '__version__',
    
    # CLV Calculator
    'CLVCalculator',
    'ClosingLineTracker',
    'CLVTier',
    'BookmakerSharpness',
    'ClosingLine',
    'CLVResult',
    'CLVPerformance',
    
    # Kelly Criterion
    'KellyCriterionCalculator',
    'BankrollManager',
    'KellyStrategy',
    'RiskProfile',
    'BetSizing',
    'BankrollState',
    'SimultaneousBetAnalysis',
    
    # Auto-Grader
    'AutoGrader',
    'GradeResult',
    'BetType',
    'GameResult',
    'GradedPrediction',
    'GradingReport',
    'PerformanceMetrics',
    
    # Line Movement
    'LineMovementAnalyzer',
    'LineMovementType',
    'MarketSentiment',
    'AlertSeverity',
    'OddsSnapshot',
    'LineMovement',
    'SharpIndicator',
    'MarketAlert',
    'MarketAnalysis',
]


# Factory functions for easy initialization

def create_betting_system(
    initial_bankroll: float = 10000.0,
    kelly_strategy: str = 'quarter',
    risk_profile: str = 'moderate',
) -> dict:
    """
    Create a complete betting system with all components.
    
    Args:
        initial_bankroll: Starting bankroll
        kelly_strategy: Kelly strategy ('full', 'three_quarter', 'half', 'quarter')
        risk_profile: Risk profile ('aggressive', 'moderate', 'conservative')
        
    Returns:
        Dictionary with all betting components
    """
    # Map strings to enums
    strategy_map = {
        'full': KellyStrategy.FULL,
        'three_quarter': KellyStrategy.THREE_QUARTER,
        'half': KellyStrategy.HALF,
        'quarter': KellyStrategy.QUARTER,
        'dynamic': KellyStrategy.DYNAMIC,
    }
    
    profile_map = {
        'aggressive': RiskProfile.AGGRESSIVE,
        'moderate': RiskProfile.MODERATE,
        'conservative': RiskProfile.CONSERVATIVE,
    }
    
    return {
        'clv_calculator': CLVCalculator(),
        'closing_line_tracker': ClosingLineTracker(),
        'kelly_calculator': KellyCriterionCalculator(
            default_strategy=strategy_map.get(kelly_strategy, KellyStrategy.QUARTER),
            risk_profile=profile_map.get(risk_profile, RiskProfile.MODERATE),
        ),
        'bankroll_manager': BankrollManager(initial_bankroll=initial_bankroll),
        'auto_grader': AutoGrader(),
        'line_analyzer': LineMovementAnalyzer(),
    }


def create_clv_tracker() -> tuple:
    """Create CLV calculator and closing line tracker."""
    return CLVCalculator(), ClosingLineTracker()


def create_kelly_system(
    bankroll: float,
    strategy: str = 'quarter',
    profile: str = 'moderate',
) -> tuple:
    """Create Kelly calculator and bankroll manager."""
    strategy_map = {
        'full': KellyStrategy.FULL,
        'three_quarter': KellyStrategy.THREE_QUARTER,
        'half': KellyStrategy.HALF,
        'quarter': KellyStrategy.QUARTER,
    }
    
    profile_map = {
        'aggressive': RiskProfile.AGGRESSIVE,
        'moderate': RiskProfile.MODERATE,
        'conservative': RiskProfile.CONSERVATIVE,
    }
    
    kelly = KellyCriterionCalculator(
        default_strategy=strategy_map.get(strategy, KellyStrategy.QUARTER),
        risk_profile=profile_map.get(profile, RiskProfile.MODERATE),
    )
    
    bankroll_mgr = BankrollManager(initial_bankroll=bankroll)
    
    return kelly, bankroll_mgr
