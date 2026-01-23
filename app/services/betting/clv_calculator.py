"""
LOYALEY - Advanced CLV (Closing Line Value) Calculator
Phase 2: Enterprise-Grade Betting Analytics

CLV is the most important metric for long-term betting success.
This module tracks and analyzes CLV performance against sharp benchmarks.
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

class CLVTier(Enum):
    """CLV performance classification."""
    ELITE = 'elite'           # +3% or better
    PROFESSIONAL = 'professional'  # +2% to +3%
    COMPETENT = 'competent'   # +1% to +2%
    BREAKEVEN = 'breakeven'   # 0% to +1%
    NEGATIVE = 'negative'     # Below 0%
    
    @property
    def description(self) -> str:
        descriptions = {
            'elite': 'Elite sharp bettor level',
            'professional': 'Professional-grade edge',
            'competent': 'Solid edge over market',
            'breakeven': 'Marginal edge, needs improvement',
            'negative': 'Losing to market - requires analysis',
        }
        return descriptions.get(self.value, '')
    
    @classmethod
    def from_clv(cls, clv: float) -> 'CLVTier':
        """Classify CLV into tier."""
        if clv >= 0.03:
            return cls.ELITE
        elif clv >= 0.02:
            return cls.PROFESSIONAL
        elif clv >= 0.01:
            return cls.COMPETENT
        elif clv >= 0.0:
            return cls.BREAKEVEN
        else:
            return cls.NEGATIVE


class BookmakerSharpness(Enum):
    """Bookmaker sharpness ranking."""
    PINNACLE = 'pinnacle'       # Sharpest - benchmark
    CIRCA = 'circa'             # Very sharp
    BOOKMAKER = 'bookmaker'     # Sharp
    BETCRIS = 'betcris'         # Sharp
    DRAFTKINGS = 'draftkings'   # Moderate
    FANDUEL = 'fanduel'         # Moderate
    BETMGM = 'betmgm'           # Square
    CAESARS = 'caesars'         # Square
    
    @property
    def vig_estimate(self) -> float:
        """Estimated vig percentage."""
        vigs = {
            'pinnacle': 0.02,
            'circa': 0.025,
            'bookmaker': 0.03,
            'betcris': 0.03,
            'draftkings': 0.045,
            'fanduel': 0.045,
            'betmgm': 0.05,
            'caesars': 0.05,
        }
        return vigs.get(self.value, 0.05)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ClosingLine:
    """Closing line data from a bookmaker."""
    game_id: str
    bet_type: str  # spread, moneyline, total
    
    # Line info
    line: float  # Spread or total number
    home_odds: int  # American odds
    away_odds: int
    
    # For totals
    over_odds: Optional[int] = None
    under_odds: Optional[int] = None
    
    # Metadata
    bookmaker: str = 'pinnacle'
    recorded_at: datetime = field(default_factory=datetime.utcnow)
    is_closing: bool = True
    
    def get_implied_probability(self, side: str) -> float:
        """Get vig-adjusted implied probability for a side."""
        if self.bet_type == 'total':
            if side == 'over':
                return self._american_to_prob_adjusted(
                    self.over_odds or -110, self.under_odds or -110
                )
            else:
                return self._american_to_prob_adjusted(
                    self.under_odds or -110, self.over_odds or -110
                )
        else:
            if side == 'home':
                return self._american_to_prob_adjusted(
                    self.home_odds, self.away_odds
                )
            else:
                return self._american_to_prob_adjusted(
                    self.away_odds, self.home_odds
                )
    
    def _american_to_prob_adjusted(self, odds: int, opposite: int) -> float:
        """Convert American odds to probability with vig removal."""
        prob = self._american_to_prob(odds)
        opp_prob = self._american_to_prob(opposite)
        total = prob + opp_prob
        if total > 0:
            return prob / total
        return 0.5
    
    def _american_to_prob(self, odds: int) -> float:
        """Convert American odds to raw implied probability."""
        if odds < 0:
            return abs(odds) / (abs(odds) + 100)
        else:
            return 100 / (odds + 100)


@dataclass
class CLVResult:
    """CLV calculation result for a single bet."""
    prediction_id: str
    game_id: str
    sport: str
    bet_type: str
    predicted_side: str
    
    # Bet timing
    bet_line: float
    bet_odds: int
    bet_probability: float  # Model probability at bet time
    bet_timestamp: datetime
    
    # Closing line
    closing_line: float
    closing_odds: int
    closing_probability: float
    closing_timestamp: datetime
    
    # CLV calculation
    line_clv: float  # CLV from line movement
    odds_clv: float  # CLV from odds movement
    combined_clv: float  # Total CLV
    
    # Classification
    clv_tier: CLVTier = field(default=CLVTier.BREAKEVEN)
    
    # Actual result
    result: Optional[str] = None  # win, loss, push
    profit_loss: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'prediction_id': self.prediction_id,
            'game_id': self.game_id,
            'sport': self.sport,
            'bet_type': self.bet_type,
            'predicted_side': self.predicted_side,
            'bet_line': self.bet_line,
            'bet_odds': self.bet_odds,
            'closing_line': self.closing_line,
            'closing_odds': self.closing_odds,
            'line_clv': round(self.line_clv, 4),
            'odds_clv': round(self.odds_clv, 4),
            'combined_clv': round(self.combined_clv, 4),
            'clv_tier': self.clv_tier.value,
            'result': self.result,
        }


@dataclass
class CLVPerformance:
    """Aggregated CLV performance statistics."""
    period: str  # 'daily', 'weekly', 'monthly', 'all_time'
    start_date: datetime
    end_date: datetime
    
    # Sample info
    total_bets: int = 0
    graded_bets: int = 0
    
    # CLV metrics
    total_clv: float = 0.0
    avg_clv: float = 0.0
    median_clv: float = 0.0
    std_clv: float = 0.0
    
    # Tier distribution
    elite_count: int = 0
    professional_count: int = 0
    competent_count: int = 0
    breakeven_count: int = 0
    negative_count: int = 0
    
    # CLV vs actual correlation
    clv_win_correlation: float = 0.0
    positive_clv_win_rate: float = 0.0
    negative_clv_win_rate: float = 0.0
    
    # By sport/type
    clv_by_sport: Dict[str, float] = field(default_factory=dict)
    clv_by_bet_type: Dict[str, float] = field(default_factory=dict)
    clv_by_tier: Dict[str, float] = field(default_factory=dict)
    
    # Performance tier
    overall_tier: CLVTier = field(default=CLVTier.BREAKEVEN)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'period': self.period,
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'total_bets': self.total_bets,
            'graded_bets': self.graded_bets,
            'avg_clv': round(self.avg_clv, 4),
            'median_clv': round(self.median_clv, 4),
            'overall_tier': self.overall_tier.value,
            'tier_distribution': {
                'elite': self.elite_count,
                'professional': self.professional_count,
                'competent': self.competent_count,
                'breakeven': self.breakeven_count,
                'negative': self.negative_count,
            },
            'positive_clv_win_rate': round(self.positive_clv_win_rate, 4),
            'negative_clv_win_rate': round(self.negative_clv_win_rate, 4),
            'clv_by_sport': {k: round(v, 4) for k, v in self.clv_by_sport.items()},
            'clv_by_bet_type': {k: round(v, 4) for k, v in self.clv_by_bet_type.items()},
        }


# =============================================================================
# CLV CALCULATOR
# =============================================================================

class CLVCalculator:
    """
    Advanced CLV calculation and tracking system.
    
    Features:
    - Pinnacle benchmark comparison
    - Line CLV and odds CLV separation
    - Vig-adjusted calculations
    - Sport-specific analysis
    - Tier-based performance tracking
    - Historical trend analysis
    """
    
    def __init__(
        self,
        benchmark_book: str = 'pinnacle',
        use_no_vig: bool = True,
    ):
        """
        Initialize CLV calculator.
        
        Args:
            benchmark_book: Bookmaker to use as sharp benchmark
            use_no_vig: Whether to remove vig from calculations
        """
        self.benchmark_book = benchmark_book
        self.use_no_vig = use_no_vig
        
        # Store CLV results
        self._clv_history: List[CLVResult] = []
        self._performance_cache: Dict[str, CLVPerformance] = {}
        
        logger.info(f"CLV Calculator initialized with {benchmark_book} benchmark")
    
    def calculate_clv(
        self,
        prediction_id: str,
        game_id: str,
        sport: str,
        bet_type: str,
        predicted_side: str,
        bet_line: float,
        bet_odds: int,
        bet_probability: float,
        bet_timestamp: datetime,
        closing_line: ClosingLine,
    ) -> CLVResult:
        """
        Calculate CLV for a single prediction.
        
        Args:
            prediction_id: Unique prediction identifier
            game_id: Game identifier
            sport: Sport code
            bet_type: Type of bet (spread, moneyline, total)
            predicted_side: Side bet was placed on
            bet_line: Line at time of bet
            bet_odds: Odds at time of bet
            bet_probability: Model probability at bet time
            bet_timestamp: When bet was placed
            closing_line: Closing line data
            
        Returns:
            CLVResult with calculated CLV metrics
        """
        # Get closing line values
        if bet_type == 'total':
            closing_line_value = closing_line.line
            if predicted_side == 'over':
                closing_odds = closing_line.over_odds or -110
            else:
                closing_odds = closing_line.under_odds or -110
        else:
            closing_line_value = closing_line.line
            if predicted_side == 'home':
                closing_odds = closing_line.home_odds
            else:
                closing_odds = closing_line.away_odds
        
        # Calculate line CLV
        line_clv = self._calculate_line_clv(
            bet_type, predicted_side, bet_line, closing_line_value
        )
        
        # Calculate odds CLV
        odds_clv = self._calculate_odds_clv(
            bet_odds, closing_odds
        )
        
        # Combined CLV (weighted average)
        # Line CLV is typically more important for spreads/totals
        # Odds CLV is more important for moneylines
        if bet_type == 'moneyline':
            combined_clv = odds_clv  # Moneyline only has odds
        else:
            combined_clv = (line_clv * 0.7) + (odds_clv * 0.3)
        
        # Determine tier
        clv_tier = CLVTier.from_clv(combined_clv)
        
        # Get closing probability
        closing_probability = closing_line.get_implied_probability(predicted_side)
        
        result = CLVResult(
            prediction_id=prediction_id,
            game_id=game_id,
            sport=sport,
            bet_type=bet_type,
            predicted_side=predicted_side,
            bet_line=bet_line,
            bet_odds=bet_odds,
            bet_probability=bet_probability,
            bet_timestamp=bet_timestamp,
            closing_line=closing_line_value,
            closing_odds=closing_odds,
            closing_probability=closing_probability,
            closing_timestamp=closing_line.recorded_at,
            line_clv=line_clv,
            odds_clv=odds_clv,
            combined_clv=combined_clv,
            clv_tier=clv_tier,
        )
        
        # Store result
        self._clv_history.append(result)
        
        # Invalidate cache
        self._performance_cache.clear()
        
        logger.debug(
            f"CLV calculated for {prediction_id}: {combined_clv:.2%} ({clv_tier.value})"
        )
        
        return result
    
    def _calculate_line_clv(
        self,
        bet_type: str,
        side: str,
        bet_line: float,
        closing_line: float,
    ) -> float:
        """
        Calculate CLV from line movement.
        
        For spreads: Getting better numbers = positive CLV
        For totals: Getting better numbers = positive CLV
        """
        if bet_type == 'moneyline':
            return 0.0  # No line CLV for moneyline
        
        line_diff = closing_line - bet_line
        
        if bet_type == 'spread':
            if side == 'home':
                # Home favored: lower spread is better (more points)
                # Home underdog: higher spread is better (more cushion)
                return -line_diff * 0.02  # ~2% per point of CLV
            else:
                return line_diff * 0.02
        elif bet_type == 'total':
            if side == 'over':
                # Over: lower line is better
                return -line_diff * 0.02
            else:
                # Under: higher line is better
                return line_diff * 0.02
        
        return 0.0
    
    def _calculate_odds_clv(
        self,
        bet_odds: int,
        closing_odds: int,
    ) -> float:
        """
        Calculate CLV from odds movement.
        
        Getting better odds than closing = positive CLV
        """
        bet_prob = self._american_to_prob(bet_odds)
        closing_prob = self._american_to_prob(closing_odds)
        
        if self.use_no_vig:
            # Assume standard -110 on other side for vig removal
            bet_prob_adj = bet_prob / (bet_prob + 0.4762)  # 0.4762 = implied prob of -110
            closing_prob_adj = closing_prob / (closing_prob + 0.4762)
            return closing_prob_adj - bet_prob_adj
        else:
            return closing_prob - bet_prob
    
    def _american_to_prob(self, odds: int) -> float:
        """Convert American odds to implied probability."""
        if odds < 0:
            return abs(odds) / (abs(odds) + 100)
        else:
            return 100 / (odds + 100)
    
    def update_result(
        self,
        prediction_id: str,
        result: str,
        profit_loss: float,
    ) -> Optional[CLVResult]:
        """
        Update CLV result with actual bet result.
        
        Args:
            prediction_id: Prediction to update
            result: 'win', 'loss', or 'push'
            profit_loss: Actual profit/loss
            
        Returns:
            Updated CLVResult or None if not found
        """
        for clv_result in self._clv_history:
            if clv_result.prediction_id == prediction_id:
                clv_result.result = result
                clv_result.profit_loss = profit_loss
                self._performance_cache.clear()
                return clv_result
        return None
    
    def get_performance(
        self,
        period: str = 'all_time',
        sport: Optional[str] = None,
        bet_type: Optional[str] = None,
        signal_tier: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> CLVPerformance:
        """
        Get aggregated CLV performance statistics.
        
        Args:
            period: 'daily', 'weekly', 'monthly', 'all_time'
            sport: Filter by sport
            bet_type: Filter by bet type
            signal_tier: Filter by prediction tier
            start_date: Start of period
            end_date: End of period
            
        Returns:
            CLVPerformance statistics
        """
        # Calculate date range
        if end_date is None:
            end_date = datetime.utcnow()
        
        if start_date is None:
            if period == 'daily':
                start_date = end_date - timedelta(days=1)
            elif period == 'weekly':
                start_date = end_date - timedelta(days=7)
            elif period == 'monthly':
                start_date = end_date - timedelta(days=30)
            else:
                start_date = datetime.min
        
        # Filter results
        filtered = [
            r for r in self._clv_history
            if start_date <= r.bet_timestamp <= end_date
            and (sport is None or r.sport == sport)
            and (bet_type is None or r.bet_type == bet_type)
        ]
        
        if not filtered:
            return CLVPerformance(
                period=period,
                start_date=start_date,
                end_date=end_date,
            )
        
        # Calculate metrics
        clv_values = [r.combined_clv for r in filtered]
        
        performance = CLVPerformance(
            period=period,
            start_date=start_date,
            end_date=end_date,
            total_bets=len(filtered),
            graded_bets=sum(1 for r in filtered if r.result is not None),
            total_clv=sum(clv_values),
            avg_clv=np.mean(clv_values),
            median_clv=np.median(clv_values),
            std_clv=np.std(clv_values) if len(clv_values) > 1 else 0.0,
        )
        
        # Tier distribution
        for r in filtered:
            if r.clv_tier == CLVTier.ELITE:
                performance.elite_count += 1
            elif r.clv_tier == CLVTier.PROFESSIONAL:
                performance.professional_count += 1
            elif r.clv_tier == CLVTier.COMPETENT:
                performance.competent_count += 1
            elif r.clv_tier == CLVTier.BREAKEVEN:
                performance.breakeven_count += 1
            else:
                performance.negative_count += 1
        
        # CLV vs win correlation
        graded = [r for r in filtered if r.result is not None]
        if graded:
            positive_clv = [r for r in graded if r.combined_clv > 0]
            negative_clv = [r for r in graded if r.combined_clv <= 0]
            
            if positive_clv:
                performance.positive_clv_win_rate = (
                    sum(1 for r in positive_clv if r.result == 'win') / 
                    len(positive_clv)
                )
            if negative_clv:
                performance.negative_clv_win_rate = (
                    sum(1 for r in negative_clv if r.result == 'win') /
                    len(negative_clv)
                )
        
        # By sport
        by_sport = defaultdict(list)
        for r in filtered:
            by_sport[r.sport].append(r.combined_clv)
        performance.clv_by_sport = {
            sport: np.mean(clvs) for sport, clvs in by_sport.items()
        }
        
        # By bet type
        by_type = defaultdict(list)
        for r in filtered:
            by_type[r.bet_type].append(r.combined_clv)
        performance.clv_by_bet_type = {
            bt: np.mean(clvs) for bt, clvs in by_type.items()
        }
        
        # Overall tier
        performance.overall_tier = CLVTier.from_clv(performance.avg_clv)
        
        return performance
    
    def get_clv_trend(
        self,
        days: int = 30,
        sport: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get daily CLV trend over time.
        
        Args:
            days: Number of days to analyze
            sport: Optional sport filter
            
        Returns:
            List of daily CLV data points
        """
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        # Filter results
        filtered = [
            r for r in self._clv_history
            if start_date <= r.bet_timestamp <= end_date
            and (sport is None or r.sport == sport)
        ]
        
        # Group by date
        by_date = defaultdict(list)
        for r in filtered:
            date_key = r.bet_timestamp.strftime('%Y-%m-%d')
            by_date[date_key].append(r.combined_clv)
        
        # Create trend data
        trend = []
        current = start_date
        cumulative_clv = 0.0
        
        while current <= end_date:
            date_key = current.strftime('%Y-%m-%d')
            day_clvs = by_date.get(date_key, [])
            
            if day_clvs:
                day_avg = np.mean(day_clvs)
                cumulative_clv += sum(day_clvs)
            else:
                day_avg = None
            
            trend.append({
                'date': date_key,
                'avg_clv': day_avg,
                'cumulative_clv': cumulative_clv,
                'bet_count': len(day_clvs),
            })
            
            current += timedelta(days=1)
        
        return trend
    
    def get_leaderboard(
        self,
        top_n: int = 10,
        by: str = 'clv',  # 'clv', 'count', 'roi'
    ) -> List[Dict[str, Any]]:
        """
        Get leaderboard of best performing categories.
        
        Args:
            top_n: Number of entries to return
            by: Metric to rank by
            
        Returns:
            List of top performers
        """
        # Group by sport + bet_type
        categories = defaultdict(lambda: {'clvs': [], 'results': []})
        
        for r in self._clv_history:
            key = f"{r.sport}_{r.bet_type}"
            categories[key]['clvs'].append(r.combined_clv)
            if r.result:
                categories[key]['results'].append(r.result)
        
        # Calculate metrics for each category
        leaderboard = []
        for key, data in categories.items():
            if len(data['clvs']) < 10:  # Minimum sample
                continue
            
            sport, bet_type = key.split('_')
            wins = sum(1 for r in data['results'] if r == 'win')
            total = len(data['results'])
            
            entry = {
                'sport': sport,
                'bet_type': bet_type,
                'total_bets': len(data['clvs']),
                'avg_clv': np.mean(data['clvs']),
                'total_clv': sum(data['clvs']),
                'win_rate': wins / total if total > 0 else 0,
                'tier': CLVTier.from_clv(np.mean(data['clvs'])).value,
            }
            leaderboard.append(entry)
        
        # Sort by metric
        if by == 'clv':
            leaderboard.sort(key=lambda x: x['avg_clv'], reverse=True)
        elif by == 'count':
            leaderboard.sort(key=lambda x: x['total_bets'], reverse=True)
        elif by == 'roi':
            leaderboard.sort(key=lambda x: x['win_rate'], reverse=True)
        
        return leaderboard[:top_n]
    
    def export_history(self) -> List[Dict[str, Any]]:
        """Export all CLV history."""
        return [r.to_dict() for r in self._clv_history]
    
    def clear_history(self):
        """Clear all stored history."""
        self._clv_history.clear()
        self._performance_cache.clear()
        logger.info("CLV history cleared")


# =============================================================================
# CLOSING LINE TRACKER
# =============================================================================

class ClosingLineTracker:
    """
    Tracks and stores closing lines from multiple bookmakers.
    
    Features:
    - Multi-bookmaker tracking
    - Pinnacle benchmark
    - Consensus line calculation
    - Historical storage
    """
    
    def __init__(self):
        """Initialize closing line tracker."""
        self._closing_lines: Dict[str, Dict[str, ClosingLine]] = {}  # game_id -> bet_type -> line
        self._history: List[ClosingLine] = []
    
    def record_closing_line(
        self,
        game_id: str,
        bet_type: str,
        line: float,
        home_odds: int,
        away_odds: int,
        over_odds: Optional[int] = None,
        under_odds: Optional[int] = None,
        bookmaker: str = 'pinnacle',
    ) -> ClosingLine:
        """
        Record a closing line for a game.
        
        Args:
            game_id: Game identifier
            bet_type: Type of bet
            line: Line value
            home_odds: Home side odds
            away_odds: Away side odds
            over_odds: Over odds (for totals)
            under_odds: Under odds (for totals)
            bookmaker: Source bookmaker
            
        Returns:
            Recorded ClosingLine
        """
        closing = ClosingLine(
            game_id=game_id,
            bet_type=bet_type,
            line=line,
            home_odds=home_odds,
            away_odds=away_odds,
            over_odds=over_odds,
            under_odds=under_odds,
            bookmaker=bookmaker,
            is_closing=True,
        )
        
        if game_id not in self._closing_lines:
            self._closing_lines[game_id] = {}
        
        self._closing_lines[game_id][bet_type] = closing
        self._history.append(closing)
        
        return closing
    
    def get_closing_line(
        self,
        game_id: str,
        bet_type: str,
    ) -> Optional[ClosingLine]:
        """Get closing line for a game and bet type."""
        return self._closing_lines.get(game_id, {}).get(bet_type)
    
    def get_all_for_game(self, game_id: str) -> Dict[str, ClosingLine]:
        """Get all closing lines for a game."""
        return self._closing_lines.get(game_id, {})


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == '__main__':
    # Create calculator
    calculator = CLVCalculator(benchmark_book='pinnacle')
    tracker = ClosingLineTracker()
    
    # Record a closing line
    closing = tracker.record_closing_line(
        game_id='NBA_20240115_LAL_GSW',
        bet_type='spread',
        line=-4.5,  # Line moved from -3.5 to -4.5
        home_odds=-110,
        away_odds=-110,
    )
    
    # Calculate CLV for a bet placed at -3.5
    clv_result = calculator.calculate_clv(
        prediction_id='pred_001',
        game_id='NBA_20240115_LAL_GSW',
        sport='NBA',
        bet_type='spread',
        predicted_side='home',
        bet_line=-3.5,
        bet_odds=-110,
        bet_probability=0.58,
        bet_timestamp=datetime.utcnow() - timedelta(hours=4),
        closing_line=closing,
    )
    
    print(f"Line CLV: {clv_result.line_clv:.2%}")
    print(f"Odds CLV: {clv_result.odds_clv:.2%}")
    print(f"Combined CLV: {clv_result.combined_clv:.2%}")
    print(f"CLV Tier: {clv_result.clv_tier.value}")
    
    # Update with result
    calculator.update_result('pred_001', 'win', profit=100)
    
    # Get performance
    perf = calculator.get_performance()
    print(f"\nOverall Performance:")
    print(f"  Average CLV: {perf.avg_clv:.2%}")
    print(f"  Overall Tier: {perf.overall_tier.value}")
