"""
ROYALEY - Auto-Grading System
Phase 2: Automatic Prediction Grading & Performance Tracking

Automatically grades predictions after games complete and tracks
detailed performance metrics across all dimensions.
"""

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import hashlib

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND TYPES
# =============================================================================

class GradeResult(Enum):
    """Possible grading results."""
    WIN = 'win'
    LOSS = 'loss'
    PUSH = 'push'
    VOID = 'void'       # Game cancelled
    PENDING = 'pending'  # Not yet graded


class BetType(Enum):
    """Bet types for grading."""
    SPREAD = 'spread'
    MONEYLINE = 'moneyline'
    TOTAL = 'total'
    FIRST_HALF_SPREAD = 'first_half_spread'
    FIRST_HALF_TOTAL = 'first_half_total'
    PLAYER_PROP = 'player_prop'


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class GameResult:
    """Final game result for grading."""
    game_id: str
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    
    # Half scores (if available)
    home_first_half: Optional[int] = None
    away_first_half: Optional[int] = None
    
    # Game status
    final: bool = True
    overtime: bool = False
    
    # Timestamps
    game_date: datetime = field(default_factory=datetime.utcnow)
    finalized_at: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def total_score(self) -> int:
        return self.home_score + self.away_score
    
    @property
    def margin(self) -> int:
        """Home team margin (positive = home win)."""
        return self.home_score - self.away_score
    
    @property
    def first_half_total(self) -> Optional[int]:
        if self.home_first_half is not None and self.away_first_half is not None:
            return self.home_first_half + self.away_first_half
        return None
    
    @property
    def first_half_margin(self) -> Optional[int]:
        if self.home_first_half is not None and self.away_first_half is not None:
            return self.home_first_half - self.away_first_half
        return None


@dataclass
class GradedPrediction:
    """A graded prediction with result and profit/loss."""
    prediction_id: str
    game_id: str
    sport: str
    
    # Bet details
    bet_type: BetType
    predicted_side: str  # 'home', 'away', 'over', 'under'
    line: float
    odds: int
    probability: float
    signal_tier: str
    
    # Grading
    result: GradeResult
    actual_value: float  # Actual margin, total, etc.
    covered_by: float    # How much we won/lost by
    
    # Profit/Loss
    stake: float = 100.0  # Assumed unit
    profit_loss: float = 0.0
    roi: float = 0.0
    
    # CLV
    closing_line: Optional[float] = None
    clv: Optional[float] = None
    
    # Timestamps
    prediction_created_at: datetime = field(default_factory=datetime.utcnow)
    graded_at: datetime = field(default_factory=datetime.utcnow)
    
    # Hash verification
    hash_verified: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'prediction_id': self.prediction_id,
            'game_id': self.game_id,
            'sport': self.sport,
            'bet_type': self.bet_type.value,
            'predicted_side': self.predicted_side,
            'line': self.line,
            'odds': self.odds,
            'probability': round(self.probability, 4),
            'signal_tier': self.signal_tier,
            'result': self.result.value,
            'actual_value': self.actual_value,
            'covered_by': round(self.covered_by, 2),
            'profit_loss': round(self.profit_loss, 2),
            'roi': round(self.roi, 4),
            'clv': round(self.clv, 4) if self.clv else None,
            'graded_at': self.graded_at.isoformat(),
        }


@dataclass
class GradingReport:
    """Summary report of grading session."""
    report_id: str
    period: str  # 'daily', 'weekly', etc.
    start_date: datetime
    end_date: datetime
    
    # Counts
    total_graded: int = 0
    wins: int = 0
    losses: int = 0
    pushes: int = 0
    voids: int = 0
    
    # Performance
    win_rate: float = 0.0
    profit_loss: float = 0.0
    roi: float = 0.0
    
    # By tier
    tier_a_record: Tuple[int, int, int] = (0, 0, 0)  # W-L-P
    tier_b_record: Tuple[int, int, int] = (0, 0, 0)
    tier_c_record: Tuple[int, int, int] = (0, 0, 0)
    tier_d_record: Tuple[int, int, int] = (0, 0, 0)
    
    # By sport
    by_sport: Dict[str, Dict] = field(default_factory=dict)
    
    # By bet type
    by_bet_type: Dict[str, Dict] = field(default_factory=dict)
    
    # CLV
    avg_clv: float = 0.0
    positive_clv_rate: float = 0.0
    
    # Detailed results
    graded_predictions: List[GradedPrediction] = field(default_factory=list)


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    # Record
    total_bets: int = 0
    wins: int = 0
    losses: int = 0
    pushes: int = 0
    
    # Financial
    total_wagered: float = 0.0
    total_won: float = 0.0
    total_lost: float = 0.0
    net_profit: float = 0.0
    roi: float = 0.0
    
    # Win rates
    overall_win_rate: float = 0.0
    spread_win_rate: float = 0.0
    moneyline_win_rate: float = 0.0
    total_win_rate: float = 0.0
    
    # Tier performance
    tier_a_win_rate: float = 0.0
    tier_b_win_rate: float = 0.0
    tier_c_win_rate: float = 0.0
    
    # CLV
    avg_clv: float = 0.0
    positive_clv_rate: float = 0.0
    
    # Streaks
    current_streak: int = 0
    max_win_streak: int = 0
    max_loss_streak: int = 0
    
    # By sport
    by_sport: Dict[str, Dict] = field(default_factory=dict)


# =============================================================================
# AUTO-GRADER
# =============================================================================

class AutoGrader:
    """
    Automatic prediction grading system.
    
    Features:
    - Grades spread, moneyline, and total bets
    - Handles pushes and half-point lines
    - Tracks CLV
    - Comprehensive performance analytics
    - Hash verification for integrity
    """
    
    def __init__(
        self,
        default_stake: float = 100.0,
    ):
        """
        Initialize auto-grader.
        
        Args:
            default_stake: Default stake for P/L calculations
        """
        self.default_stake = default_stake
        
        # Storage
        self._graded: List[GradedPrediction] = []
        self._pending: Dict[str, Dict] = {}  # prediction_id -> prediction data
        
        logger.info("AutoGrader initialized")
    
    def add_pending_prediction(
        self,
        prediction_id: str,
        game_id: str,
        sport: str,
        bet_type: str,
        predicted_side: str,
        line: float,
        odds: int,
        probability: float,
        signal_tier: str,
        prediction_hash: str,
        stake: Optional[float] = None,
        closing_line: Optional[float] = None,
    ):
        """
        Add a prediction to the pending queue for grading.
        
        Args:
            prediction_id: Unique prediction identifier
            game_id: Game identifier
            sport: Sport code
            bet_type: Type of bet
            predicted_side: Predicted side
            line: Bet line
            odds: American odds
            probability: Model probability
            signal_tier: Prediction tier
            prediction_hash: SHA-256 hash for verification
            stake: Bet stake (optional)
            closing_line: Closing line for CLV (optional)
        """
        self._pending[prediction_id] = {
            'prediction_id': prediction_id,
            'game_id': game_id,
            'sport': sport,
            'bet_type': bet_type,
            'predicted_side': predicted_side,
            'line': line,
            'odds': odds,
            'probability': probability,
            'signal_tier': signal_tier,
            'prediction_hash': prediction_hash,
            'stake': stake or self.default_stake,
            'closing_line': closing_line,
            'created_at': datetime.utcnow(),
        }
        
        logger.debug(f"Added pending prediction: {prediction_id}")
    
    def grade_game(
        self,
        game_result: GameResult,
    ) -> List[GradedPrediction]:
        """
        Grade all pending predictions for a completed game.
        
        Args:
            game_result: Final game result
            
        Returns:
            List of graded predictions
        """
        graded = []
        
        # Find all pending predictions for this game
        to_grade = [
            (pid, pred) for pid, pred in self._pending.items()
            if pred['game_id'] == game_result.game_id
        ]
        
        for prediction_id, prediction in to_grade:
            try:
                graded_pred = self._grade_single_prediction(
                    prediction, game_result
                )
                graded.append(graded_pred)
                self._graded.append(graded_pred)
                del self._pending[prediction_id]
                
                logger.info(
                    f"Graded {prediction_id}: {graded_pred.result.value} "
                    f"(P/L: ${graded_pred.profit_loss:.2f})"
                )
                
            except Exception as e:
                logger.error(f"Error grading {prediction_id}: {e}")
        
        return graded
    
    def _grade_single_prediction(
        self,
        prediction: Dict,
        game_result: GameResult,
    ) -> GradedPrediction:
        """Grade a single prediction against game result."""
        bet_type = BetType(prediction['bet_type'])
        side = prediction['predicted_side']
        line = prediction['line']
        odds = prediction['odds']
        stake = prediction['stake']
        
        # Determine result based on bet type
        if bet_type == BetType.SPREAD:
            result, actual_value, covered_by = self._grade_spread(
                side, line, game_result
            )
        elif bet_type == BetType.MONEYLINE:
            result, actual_value, covered_by = self._grade_moneyline(
                side, game_result
            )
        elif bet_type == BetType.TOTAL:
            result, actual_value, covered_by = self._grade_total(
                side, line, game_result
            )
        elif bet_type == BetType.FIRST_HALF_SPREAD:
            result, actual_value, covered_by = self._grade_first_half_spread(
                side, line, game_result
            )
        elif bet_type == BetType.FIRST_HALF_TOTAL:
            result, actual_value, covered_by = self._grade_first_half_total(
                side, line, game_result
            )
        else:
            result = GradeResult.VOID
            actual_value = 0.0
            covered_by = 0.0
        
        # Calculate profit/loss
        profit_loss = self._calculate_profit_loss(result, stake, odds)
        roi = profit_loss / stake if stake > 0 else 0.0
        
        # Calculate CLV
        clv = None
        if prediction.get('closing_line') is not None:
            clv = self._calculate_clv(
                bet_type, side, line, prediction['closing_line']
            )
        
        return GradedPrediction(
            prediction_id=prediction['prediction_id'],
            game_id=prediction['game_id'],
            sport=prediction['sport'],
            bet_type=bet_type,
            predicted_side=side,
            line=line,
            odds=odds,
            probability=prediction['probability'],
            signal_tier=prediction['signal_tier'],
            result=result,
            actual_value=actual_value,
            covered_by=covered_by,
            stake=stake,
            profit_loss=profit_loss,
            roi=roi,
            closing_line=prediction.get('closing_line'),
            clv=clv,
            prediction_created_at=prediction.get('created_at', datetime.utcnow()),
            graded_at=datetime.utcnow(),
        )
    
    def _grade_spread(
        self,
        side: str,
        line: float,
        game: GameResult,
    ) -> Tuple[GradeResult, float, float]:
        """
        Grade a spread bet.
        
        For home spread: home_score + line > away_score = WIN
        For away spread: away_score - line > home_score = WIN
        """
        actual_margin = game.margin  # home - away
        
        if side == 'home':
            # Home team spread: need home_score + spread > away_score
            adjusted_margin = actual_margin + line
            covered_by = adjusted_margin
        else:  # away
            # Away team spread: need away_score - spread > home_score
            adjusted_margin = -actual_margin + line
            covered_by = adjusted_margin
        
        if adjusted_margin > 0:
            result = GradeResult.WIN
        elif adjusted_margin < 0:
            result = GradeResult.LOSS
        else:
            result = GradeResult.PUSH
        
        return result, actual_margin, covered_by
    
    def _grade_moneyline(
        self,
        side: str,
        game: GameResult,
    ) -> Tuple[GradeResult, float, float]:
        """Grade a moneyline bet."""
        actual_margin = game.margin
        
        if actual_margin > 0:
            winner = 'home'
        elif actual_margin < 0:
            winner = 'away'
        else:
            return GradeResult.PUSH, actual_margin, 0.0
        
        if side == winner:
            result = GradeResult.WIN
            covered_by = abs(actual_margin)
        else:
            result = GradeResult.LOSS
            covered_by = -abs(actual_margin)
        
        return result, actual_margin, covered_by
    
    def _grade_total(
        self,
        side: str,
        line: float,
        game: GameResult,
    ) -> Tuple[GradeResult, float, float]:
        """Grade a total (over/under) bet."""
        actual_total = game.total_score
        difference = actual_total - line
        
        if side == 'over':
            if difference > 0:
                result = GradeResult.WIN
                covered_by = difference
            elif difference < 0:
                result = GradeResult.LOSS
                covered_by = difference
            else:
                result = GradeResult.PUSH
                covered_by = 0.0
        else:  # under
            if difference < 0:
                result = GradeResult.WIN
                covered_by = -difference
            elif difference > 0:
                result = GradeResult.LOSS
                covered_by = -difference
            else:
                result = GradeResult.PUSH
                covered_by = 0.0
        
        return result, actual_total, covered_by
    
    def _grade_first_half_spread(
        self,
        side: str,
        line: float,
        game: GameResult,
    ) -> Tuple[GradeResult, float, float]:
        """Grade a first half spread bet."""
        if game.first_half_margin is None:
            return GradeResult.VOID, 0.0, 0.0
        
        actual_margin = game.first_half_margin
        
        if side == 'home':
            adjusted_margin = actual_margin + line
            covered_by = adjusted_margin
        else:
            adjusted_margin = -actual_margin + line
            covered_by = adjusted_margin
        
        if adjusted_margin > 0:
            result = GradeResult.WIN
        elif adjusted_margin < 0:
            result = GradeResult.LOSS
        else:
            result = GradeResult.PUSH
        
        return result, actual_margin, covered_by
    
    def _grade_first_half_total(
        self,
        side: str,
        line: float,
        game: GameResult,
    ) -> Tuple[GradeResult, float, float]:
        """Grade a first half total bet."""
        if game.first_half_total is None:
            return GradeResult.VOID, 0.0, 0.0
        
        actual_total = game.first_half_total
        difference = actual_total - line
        
        if side == 'over':
            if difference > 0:
                result = GradeResult.WIN
                covered_by = difference
            elif difference < 0:
                result = GradeResult.LOSS
                covered_by = difference
            else:
                result = GradeResult.PUSH
                covered_by = 0.0
        else:
            if difference < 0:
                result = GradeResult.WIN
                covered_by = -difference
            elif difference > 0:
                result = GradeResult.LOSS
                covered_by = -difference
            else:
                result = GradeResult.PUSH
                covered_by = 0.0
        
        return result, actual_total, covered_by
    
    def _calculate_profit_loss(
        self,
        result: GradeResult,
        stake: float,
        odds: int,
    ) -> float:
        """Calculate profit/loss for a result."""
        if result == GradeResult.PUSH or result == GradeResult.VOID:
            return 0.0
        
        if result == GradeResult.WIN:
            if odds < 0:
                return stake * (100 / abs(odds))
            else:
                return stake * (odds / 100)
        else:  # LOSS
            return -stake
    
    def _calculate_clv(
        self,
        bet_type: BetType,
        side: str,
        bet_line: float,
        closing_line: float,
    ) -> float:
        """Calculate CLV from line movement."""
        if bet_type == BetType.MONEYLINE:
            return 0.0  # Different calculation for ML
        
        line_diff = closing_line - bet_line
        
        if bet_type in [BetType.SPREAD, BetType.FIRST_HALF_SPREAD]:
            if side == 'home':
                return -line_diff * 0.02  # Per point
            else:
                return line_diff * 0.02
        elif bet_type in [BetType.TOTAL, BetType.FIRST_HALF_TOTAL]:
            if side == 'over':
                return -line_diff * 0.02
            else:
                return line_diff * 0.02
        
        return 0.0
    
    def get_performance_metrics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        sport: Optional[str] = None,
        signal_tier: Optional[str] = None,
    ) -> PerformanceMetrics:
        """
        Get comprehensive performance metrics.
        
        Args:
            start_date: Start of period
            end_date: End of period
            sport: Filter by sport
            signal_tier: Filter by tier
            
        Returns:
            PerformanceMetrics object
        """
        # Filter predictions
        filtered = self._graded
        if start_date:
            filtered = [p for p in filtered if p.graded_at >= start_date]
        if end_date:
            filtered = [p for p in filtered if p.graded_at <= end_date]
        if sport:
            filtered = [p for p in filtered if p.sport == sport]
        if signal_tier:
            filtered = [p for p in filtered if p.signal_tier == signal_tier]
        
        if not filtered:
            return PerformanceMetrics()
        
        metrics = PerformanceMetrics()
        
        # Basic counts
        metrics.total_bets = len(filtered)
        metrics.wins = sum(1 for p in filtered if p.result == GradeResult.WIN)
        metrics.losses = sum(1 for p in filtered if p.result == GradeResult.LOSS)
        metrics.pushes = sum(1 for p in filtered if p.result == GradeResult.PUSH)
        
        # Financial
        metrics.total_wagered = sum(p.stake for p in filtered)
        metrics.total_won = sum(p.profit_loss for p in filtered if p.profit_loss > 0)
        metrics.total_lost = sum(abs(p.profit_loss) for p in filtered if p.profit_loss < 0)
        metrics.net_profit = sum(p.profit_loss for p in filtered)
        metrics.roi = metrics.net_profit / metrics.total_wagered if metrics.total_wagered > 0 else 0
        
        # Win rates
        decided = [p for p in filtered if p.result in [GradeResult.WIN, GradeResult.LOSS]]
        if decided:
            metrics.overall_win_rate = sum(1 for p in decided if p.result == GradeResult.WIN) / len(decided)
        
        # By bet type
        for bt in [BetType.SPREAD, BetType.MONEYLINE, BetType.TOTAL]:
            bt_preds = [p for p in decided if p.bet_type == bt]
            if bt_preds:
                win_rate = sum(1 for p in bt_preds if p.result == GradeResult.WIN) / len(bt_preds)
                if bt == BetType.SPREAD:
                    metrics.spread_win_rate = win_rate
                elif bt == BetType.MONEYLINE:
                    metrics.moneyline_win_rate = win_rate
                elif bt == BetType.TOTAL:
                    metrics.total_win_rate = win_rate
        
        # By tier
        for tier in ['A', 'B', 'C']:
            tier_preds = [p for p in decided if p.signal_tier == tier]
            if tier_preds:
                win_rate = sum(1 for p in tier_preds if p.result == GradeResult.WIN) / len(tier_preds)
                if tier == 'A':
                    metrics.tier_a_win_rate = win_rate
                elif tier == 'B':
                    metrics.tier_b_win_rate = win_rate
                elif tier == 'C':
                    metrics.tier_c_win_rate = win_rate
        
        # CLV
        clv_preds = [p for p in filtered if p.clv is not None]
        if clv_preds:
            metrics.avg_clv = np.mean([p.clv for p in clv_preds])
            metrics.positive_clv_rate = sum(1 for p in clv_preds if p.clv > 0) / len(clv_preds)
        
        # Streaks
        streak = 0
        max_win = 0
        max_loss = 0
        current_streak = 0
        
        for p in sorted(decided, key=lambda x: x.graded_at):
            if p.result == GradeResult.WIN:
                if current_streak >= 0:
                    current_streak += 1
                else:
                    current_streak = 1
                max_win = max(max_win, current_streak)
            else:
                if current_streak <= 0:
                    current_streak -= 1
                else:
                    current_streak = -1
                max_loss = max(max_loss, abs(current_streak))
        
        metrics.current_streak = current_streak
        metrics.max_win_streak = max_win
        metrics.max_loss_streak = max_loss
        
        # By sport
        sports = set(p.sport for p in filtered)
        for s in sports:
            sport_preds = [p for p in filtered if p.sport == s]
            sport_decided = [p for p in sport_preds if p.result in [GradeResult.WIN, GradeResult.LOSS]]
            if sport_decided:
                metrics.by_sport[s] = {
                    'total': len(sport_preds),
                    'wins': sum(1 for p in sport_decided if p.result == GradeResult.WIN),
                    'losses': sum(1 for p in sport_decided if p.result == GradeResult.LOSS),
                    'win_rate': sum(1 for p in sport_decided if p.result == GradeResult.WIN) / len(sport_decided),
                    'profit': sum(p.profit_loss for p in sport_preds),
                }
        
        return metrics
    
    def generate_report(
        self,
        period: str = 'daily',
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> GradingReport:
        """
        Generate a grading summary report.
        
        Args:
            period: Report period type
            start_date: Start of period
            end_date: End of period
            
        Returns:
            GradingReport
        """
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
        
        # Filter predictions
        filtered = [
            p for p in self._graded
            if start_date <= p.graded_at <= end_date
        ]
        
        report = GradingReport(
            report_id=f"report_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            period=period,
            start_date=start_date,
            end_date=end_date,
            graded_predictions=filtered,
        )
        
        if not filtered:
            return report
        
        # Basic counts
        report.total_graded = len(filtered)
        report.wins = sum(1 for p in filtered if p.result == GradeResult.WIN)
        report.losses = sum(1 for p in filtered if p.result == GradeResult.LOSS)
        report.pushes = sum(1 for p in filtered if p.result == GradeResult.PUSH)
        report.voids = sum(1 for p in filtered if p.result == GradeResult.VOID)
        
        # Performance
        decided = report.wins + report.losses
        if decided > 0:
            report.win_rate = report.wins / decided
        report.profit_loss = sum(p.profit_loss for p in filtered)
        total_wagered = sum(p.stake for p in filtered)
        if total_wagered > 0:
            report.roi = report.profit_loss / total_wagered
        
        # By tier
        for tier in ['A', 'B', 'C', 'D']:
            tier_preds = [p for p in filtered if p.signal_tier == tier]
            w = sum(1 for p in tier_preds if p.result == GradeResult.WIN)
            l = sum(1 for p in tier_preds if p.result == GradeResult.LOSS)
            p_count = sum(1 for p in tier_preds if p.result == GradeResult.PUSH)
            record = (w, l, p_count)
            if tier == 'A':
                report.tier_a_record = record
            elif tier == 'B':
                report.tier_b_record = record
            elif tier == 'C':
                report.tier_c_record = record
            elif tier == 'D':
                report.tier_d_record = record
        
        # By sport
        sports = set(p.sport for p in filtered)
        for sport in sports:
            sport_preds = [p for p in filtered if p.sport == sport]
            sw = sum(1 for p in sport_preds if p.result == GradeResult.WIN)
            sl = sum(1 for p in sport_preds if p.result == GradeResult.LOSS)
            sp = sum(p.profit_loss for p in sport_preds)
            report.by_sport[sport] = {
                'record': f"{sw}-{sl}",
                'profit': sp,
                'win_rate': sw / (sw + sl) if (sw + sl) > 0 else 0,
            }
        
        # By bet type
        bet_types = set(p.bet_type for p in filtered)
        for bt in bet_types:
            bt_preds = [p for p in filtered if p.bet_type == bt]
            btw = sum(1 for p in bt_preds if p.result == GradeResult.WIN)
            btl = sum(1 for p in bt_preds if p.result == GradeResult.LOSS)
            btp = sum(p.profit_loss for p in bt_preds)
            report.by_bet_type[bt.value] = {
                'record': f"{btw}-{btl}",
                'profit': btp,
                'win_rate': btw / (btw + btl) if (btw + btl) > 0 else 0,
            }
        
        # CLV
        clv_preds = [p for p in filtered if p.clv is not None]
        if clv_preds:
            report.avg_clv = np.mean([p.clv for p in clv_preds])
            report.positive_clv_rate = sum(1 for p in clv_preds if p.clv > 0) / len(clv_preds)
        
        return report
    
    def get_pending_count(self) -> int:
        """Get count of pending predictions."""
        return len(self._pending)
    
    def get_pending_by_game(self, game_id: str) -> List[Dict]:
        """Get pending predictions for a game."""
        return [
            pred for pred in self._pending.values()
            if pred['game_id'] == game_id
        ]
    
    def export_all(self) -> List[Dict]:
        """Export all graded predictions."""
        return [p.to_dict() for p in self._graded]


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == '__main__':
    # Create grader
    grader = AutoGrader(default_stake=100)
    
    # Add pending prediction
    grader.add_pending_prediction(
        prediction_id='pred_001',
        game_id='NBA_20240115_LAL_GSW',
        sport='NBA',
        bet_type='spread',
        predicted_side='home',
        line=-3.5,
        odds=-110,
        probability=0.58,
        signal_tier='B',
        prediction_hash='abc123',
        closing_line=-4.5,
    )
    
    grader.add_pending_prediction(
        prediction_id='pred_002',
        game_id='NBA_20240115_LAL_GSW',
        sport='NBA',
        bet_type='total',
        predicted_side='over',
        line=220.5,
        odds=-110,
        probability=0.55,
        signal_tier='C',
        prediction_hash='def456',
        closing_line=222.5,
    )
    
    # Game finishes
    result = GameResult(
        game_id='NBA_20240115_LAL_GSW',
        home_team='Lakers',
        away_team='Warriors',
        home_score=115,
        away_score=108,
        home_first_half=58,
        away_first_half=52,
    )
    
    # Grade the game
    graded = grader.grade_game(result)
    
    print("=== Grading Results ===")
    for g in graded:
        print(f"{g.prediction_id}: {g.result.value}")
        print(f"  Actual: {g.actual_value}, Covered by: {g.covered_by}")
        print(f"  P/L: ${g.profit_loss:.2f} (ROI: {g.roi:.2%})")
        print(f"  CLV: {g.clv:.2%}" if g.clv else "  CLV: N/A")
    
    # Generate report
    report = grader.generate_report('daily')
    print(f"\n=== Daily Report ===")
    print(f"Total: {report.total_graded}")
    print(f"Record: {report.wins}-{report.losses}-{report.pushes}")
    print(f"Win Rate: {report.win_rate:.1%}")
    print(f"P/L: ${report.profit_loss:.2f}")
    print(f"ROI: {report.roi:.2%}")
    
    # Get performance metrics
    metrics = grader.get_performance_metrics()
    print(f"\n=== Performance Metrics ===")
    print(f"Overall Win Rate: {metrics.overall_win_rate:.1%}")
    print(f"Net Profit: ${metrics.net_profit:.2f}")
