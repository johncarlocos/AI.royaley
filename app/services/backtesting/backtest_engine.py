"""
ROYALEY - Backtesting Engine
Comprehensive backtesting system for strategy evaluation
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import json
import hashlib
from decimal import Decimal, ROUND_HALF_UP

logger = logging.getLogger(__name__)


class BetType(Enum):
    SPREAD = "spread"
    MONEYLINE = "moneyline"
    TOTAL = "total"
    PLAYER_PROP = "player_prop"


class BetResult(Enum):
    WIN = "win"
    LOSS = "loss"
    PUSH = "push"
    PENDING = "pending"


@dataclass
class BacktestConfig:
    """Configuration for backtesting run"""
    name: str
    start_date: datetime
    end_date: datetime
    initial_bankroll: float = 10000.0
    kelly_fraction: float = 0.25
    max_bet_percent: float = 0.02
    min_edge_threshold: float = 0.03
    min_probability: float = 0.55
    sports: List[str] = field(default_factory=lambda: ["NFL", "NBA", "MLB", "NHL"])
    bet_types: List[str] = field(default_factory=lambda: ["spread", "moneyline", "total"])
    signal_tiers: List[str] = field(default_factory=lambda: ["A", "B"])
    use_closing_line: bool = False
    include_juice: bool = True
    max_daily_bets: Optional[int] = None
    max_concurrent_bets: Optional[int] = None
    stop_loss_percent: Optional[float] = None
    take_profit_percent: Optional[float] = None


@dataclass
class BacktestBet:
    """Individual bet in backtest"""
    id: str
    date: datetime
    sport: str
    game_id: str
    bet_type: BetType
    predicted_side: str
    probability: float
    edge: float
    line: float
    odds: int
    stake: float
    signal_tier: str
    result: BetResult = BetResult.PENDING
    profit_loss: float = 0.0
    closing_line: Optional[float] = None
    clv: Optional[float] = None
    actual_outcome: Optional[str] = None
    settled_at: Optional[datetime] = None


@dataclass
class DailySnapshot:
    """Daily bankroll snapshot"""
    date: datetime
    starting_balance: float
    ending_balance: float
    bets_placed: int
    bets_won: int
    bets_lost: int
    bets_pushed: int
    total_wagered: float
    total_profit_loss: float
    roi_percent: float
    win_rate: float
    avg_odds: float
    avg_edge: float


@dataclass
class BacktestResult:
    """Results from backtesting run"""
    config: BacktestConfig
    run_id: str
    started_at: datetime
    completed_at: datetime
    
    # Overall metrics
    initial_bankroll: float
    final_bankroll: float
    total_profit_loss: float
    total_roi_percent: float
    
    # Betting metrics
    total_bets: int
    bets_won: int
    bets_lost: int
    bets_pushed: int
    win_rate: float
    
    # Risk metrics
    max_drawdown_percent: float
    max_drawdown_amount: float
    max_runup_percent: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # Profit metrics
    total_wagered: float
    avg_bet_size: float
    avg_odds: int
    avg_edge: float
    profit_factor: float
    expected_value: float
    
    # CLV metrics
    avg_clv: float
    clv_positive_rate: float
    
    # Breakdowns
    by_sport: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    by_bet_type: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    by_tier: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    by_month: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Time series
    daily_snapshots: List[DailySnapshot] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    drawdown_curve: List[float] = field(default_factory=list)
    
    # All bets
    bets: List[BacktestBet] = field(default_factory=list)


class BacktestEngine:
    """
    Main backtesting engine for evaluating betting strategies
    """
    
    def __init__(
        self,
        prediction_source: Callable[[datetime, datetime], List[Dict]],
        odds_source: Callable[[str], Dict],
        results_source: Callable[[str], Dict],
        closing_line_source: Optional[Callable[[str], Dict]] = None
    ):
        """
        Initialize backtest engine with data sources
        
        Args:
            prediction_source: Function to get predictions for date range
            odds_source: Function to get odds for a game
            results_source: Function to get game results
            closing_line_source: Optional function to get closing lines
        """
        self.prediction_source = prediction_source
        self.odds_source = odds_source
        self.results_source = results_source
        self.closing_line_source = closing_line_source
        
        self._current_bankroll = 0.0
        self._peak_bankroll = 0.0
        self._trough_bankroll = 0.0
        self._pending_bets: List[BacktestBet] = []
        self._settled_bets: List[BacktestBet] = []
        self._daily_returns: List[float] = []
        
    def run(self, config: BacktestConfig) -> BacktestResult:
        """
        Execute backtest with given configuration
        
        Args:
            config: Backtest configuration
            
        Returns:
            BacktestResult with comprehensive metrics
        """
        logger.info(f"Starting backtest: {config.name}")
        started_at = datetime.now()
        run_id = self._generate_run_id(config)
        
        # Initialize state
        self._current_bankroll = config.initial_bankroll
        self._peak_bankroll = config.initial_bankroll
        self._trough_bankroll = config.initial_bankroll
        self._pending_bets = []
        self._settled_bets = []
        self._daily_returns = []
        
        daily_snapshots = []
        equity_curve = [config.initial_bankroll]
        drawdown_curve = [0.0]
        
        # Get all predictions for date range
        predictions = self.prediction_source(config.start_date, config.end_date)
        logger.info(f"Loaded {len(predictions)} predictions")
        
        # Group predictions by date
        predictions_by_date = self._group_by_date(predictions)
        
        # Iterate through each day
        current_date = config.start_date
        while current_date <= config.end_date:
            day_start_balance = self._current_bankroll
            
            # Check stop loss / take profit
            if self._check_stop_conditions(config, day_start_balance):
                logger.info(f"Stop condition reached on {current_date}")
                break
            
            # Settle completed bets from previous days
            self._settle_pending_bets(current_date)
            
            # Get today's predictions
            day_predictions = predictions_by_date.get(current_date.date(), [])
            
            # Filter predictions based on config
            filtered = self._filter_predictions(day_predictions, config)
            
            # Place bets
            bets_placed_today = 0
            for pred in filtered:
                if config.max_daily_bets and bets_placed_today >= config.max_daily_bets:
                    break
                    
                if config.max_concurrent_bets and len(self._pending_bets) >= config.max_concurrent_bets:
                    break
                
                bet = self._create_bet(pred, config, current_date)
                if bet and bet.stake > 0:
                    self._pending_bets.append(bet)
                    self._current_bankroll -= bet.stake
                    bets_placed_today += 1
            
            # Calculate daily metrics
            day_end_balance = self._current_bankroll + sum(b.stake for b in self._pending_bets)
            day_profit = day_end_balance - day_start_balance
            day_return = day_profit / day_start_balance if day_start_balance > 0 else 0
            self._daily_returns.append(day_return)
            
            # Update peak/trough
            if day_end_balance > self._peak_bankroll:
                self._peak_bankroll = day_end_balance
            if day_end_balance < self._trough_bankroll:
                self._trough_bankroll = day_end_balance
            
            # Calculate drawdown
            drawdown = (self._peak_bankroll - day_end_balance) / self._peak_bankroll if self._peak_bankroll > 0 else 0
            
            # Record daily snapshot
            settled_today = [b for b in self._settled_bets if b.settled_at and b.settled_at.date() == current_date.date()]
            snapshot = self._create_daily_snapshot(
                current_date, day_start_balance, day_end_balance,
                bets_placed_today, settled_today
            )
            daily_snapshots.append(snapshot)
            equity_curve.append(day_end_balance)
            drawdown_curve.append(drawdown)
            
            current_date += timedelta(days=1)
        
        # Settle any remaining pending bets
        self._settle_all_pending()
        
        # Calculate final metrics
        completed_at = datetime.now()
        result = self._calculate_results(
            config, run_id, started_at, completed_at,
            daily_snapshots, equity_curve, drawdown_curve
        )
        
        logger.info(f"Backtest completed: ROI={result.total_roi_percent:.2f}%, Win Rate={result.win_rate:.2f}%")
        return result
    
    def _generate_run_id(self, config: BacktestConfig) -> str:
        """Generate unique run ID"""
        data = f"{config.name}{config.start_date}{config.end_date}{datetime.now()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def _group_by_date(self, predictions: List[Dict]) -> Dict[Any, List[Dict]]:
        """Group predictions by date"""
        by_date = {}
        for pred in predictions:
            pred_date = pred.get('date')
            if isinstance(pred_date, datetime):
                pred_date = pred_date.date()
            if pred_date not in by_date:
                by_date[pred_date] = []
            by_date[pred_date].append(pred)
        return by_date
    
    def _filter_predictions(self, predictions: List[Dict], config: BacktestConfig) -> List[Dict]:
        """Filter predictions based on config criteria"""
        filtered = []
        for pred in predictions:
            # Check sport
            if pred.get('sport') not in config.sports:
                continue
            
            # Check bet type
            if pred.get('bet_type') not in config.bet_types:
                continue
            
            # Check signal tier
            if pred.get('signal_tier') not in config.signal_tiers:
                continue
            
            # Check probability threshold
            if pred.get('probability', 0) < config.min_probability:
                continue
            
            # Check edge threshold
            if pred.get('edge', 0) < config.min_edge_threshold:
                continue
            
            filtered.append(pred)
        
        # Sort by edge (highest first)
        filtered.sort(key=lambda x: x.get('edge', 0), reverse=True)
        return filtered
    
    def _create_bet(self, prediction: Dict, config: BacktestConfig, date: datetime) -> Optional[BacktestBet]:
        """Create bet from prediction"""
        try:
            # Get odds
            game_id = prediction.get('game_id')
            odds_data = self.odds_source(game_id) if game_id else {}
            
            # Use closing line if configured
            if config.use_closing_line and self.closing_line_source:
                closing_data = self.closing_line_source(game_id)
                line = closing_data.get('line', prediction.get('line', 0))
                odds = closing_data.get('odds', prediction.get('odds', -110))
            else:
                line = prediction.get('line', 0)
                odds = prediction.get('odds', -110)
            
            probability = prediction.get('probability', 0.5)
            
            # Calculate edge
            implied_prob = self._odds_to_probability(odds)
            edge = probability - implied_prob
            
            if edge < config.min_edge_threshold:
                return None
            
            # Calculate stake using Kelly criterion
            stake = self._calculate_kelly_stake(
                probability, odds, 
                self._current_bankroll,
                config.kelly_fraction,
                config.max_bet_percent
            )
            
            if stake < 10:  # Minimum $10 bet
                return None
            
            bet_id = hashlib.sha256(f"{game_id}{prediction.get('bet_type')}{date}".encode()).hexdigest()[:12]
            
            return BacktestBet(
                id=bet_id,
                date=date,
                sport=prediction.get('sport', ''),
                game_id=game_id,
                bet_type=BetType(prediction.get('bet_type', 'spread')),
                predicted_side=prediction.get('predicted_side', ''),
                probability=probability,
                edge=edge,
                line=line,
                odds=odds,
                stake=stake,
                signal_tier=prediction.get('signal_tier', 'C'),
                closing_line=prediction.get('closing_line')
            )
        except Exception as e:
            logger.error(f"Error creating bet: {e}")
            return None
    
    def _calculate_kelly_stake(
        self, 
        probability: float, 
        odds: int, 
        bankroll: float,
        kelly_fraction: float,
        max_bet_percent: float
    ) -> float:
        """Calculate bet size using fractional Kelly criterion"""
        decimal_odds = self._american_to_decimal(odds)
        b = decimal_odds - 1
        p = probability
        q = 1 - p
        
        if b <= 0:
            return 0
        
        # Full Kelly
        full_kelly = (b * p - q) / b
        
        if full_kelly <= 0:
            return 0
        
        # Apply fractional Kelly
        kelly = full_kelly * kelly_fraction
        
        # Cap at max bet percent
        kelly = min(kelly, max_bet_percent)
        
        return round(bankroll * kelly, 2)
    
    def _american_to_decimal(self, american: int) -> float:
        """Convert American odds to decimal"""
        if american > 0:
            return 1 + (american / 100)
        else:
            return 1 + (100 / abs(american))
    
    def _odds_to_probability(self, american: int) -> float:
        """Convert American odds to implied probability"""
        if american > 0:
            return 100 / (american + 100)
        else:
            return abs(american) / (abs(american) + 100)
    
    def _settle_pending_bets(self, current_date: datetime):
        """Settle bets that have completed"""
        still_pending = []
        
        for bet in self._pending_bets:
            # Check if game has finished (assume same day for simplicity)
            if bet.date.date() < current_date.date():
                result_data = self.results_source(bet.game_id)
                
                if result_data:
                    result, profit_loss = self._grade_bet(bet, result_data)
                    bet.result = result
                    bet.profit_loss = profit_loss
                    bet.settled_at = current_date
                    bet.actual_outcome = result_data.get('outcome')
                    
                    # Calculate CLV if available
                    if self.closing_line_source:
                        closing_data = self.closing_line_source(bet.game_id)
                        if closing_data:
                            bet.closing_line = closing_data.get('line')
                            bet.clv = self._calculate_clv(bet, closing_data)
                    
                    # Update bankroll
                    if result == BetResult.WIN:
                        self._current_bankroll += bet.stake + profit_loss
                    elif result == BetResult.PUSH:
                        self._current_bankroll += bet.stake
                    # Loss: stake already deducted
                    
                    self._settled_bets.append(bet)
                else:
                    still_pending.append(bet)
            else:
                still_pending.append(bet)
        
        self._pending_bets = still_pending
    
    def _settle_all_pending(self):
        """Force settle all remaining pending bets"""
        for bet in self._pending_bets:
            result_data = self.results_source(bet.game_id)
            
            if result_data:
                result, profit_loss = self._grade_bet(bet, result_data)
                bet.result = result
                bet.profit_loss = profit_loss
                bet.settled_at = datetime.now()
                
                if result == BetResult.WIN:
                    self._current_bankroll += bet.stake + profit_loss
                elif result == BetResult.PUSH:
                    self._current_bankroll += bet.stake
                
                self._settled_bets.append(bet)
    
    def _grade_bet(self, bet: BacktestBet, result_data: Dict) -> tuple:
        """Grade a bet and calculate profit/loss"""
        home_score = result_data.get('home_score', 0)
        away_score = result_data.get('away_score', 0)
        
        if bet.bet_type == BetType.SPREAD:
            return self._grade_spread(bet, home_score, away_score)
        elif bet.bet_type == BetType.MONEYLINE:
            return self._grade_moneyline(bet, home_score, away_score)
        elif bet.bet_type == BetType.TOTAL:
            actual_total = result_data.get('total', home_score + away_score)
            return self._grade_total(bet, actual_total)
        
        return BetResult.LOSS, -bet.stake
    
    def _grade_spread(self, bet: BacktestBet, home_score: int, away_score: int) -> tuple:
        """Grade spread bet"""
        actual_margin = home_score - away_score
        
        if bet.predicted_side == 'home':
            result_margin = actual_margin + bet.line
        else:
            result_margin = -actual_margin + bet.line
        
        if result_margin > 0:
            profit = self._calculate_profit(bet.stake, bet.odds)
            return BetResult.WIN, profit
        elif result_margin < 0:
            return BetResult.LOSS, -bet.stake
        else:
            return BetResult.PUSH, 0
    
    def _grade_moneyline(self, bet: BacktestBet, home_score: int, away_score: int) -> tuple:
        """Grade moneyline bet"""
        if home_score == away_score:
            return BetResult.PUSH, 0
        
        home_won = home_score > away_score
        bet_on_home = bet.predicted_side == 'home'
        
        if (home_won and bet_on_home) or (not home_won and not bet_on_home):
            profit = self._calculate_profit(bet.stake, bet.odds)
            return BetResult.WIN, profit
        else:
            return BetResult.LOSS, -bet.stake
    
    def _grade_total(self, bet: BacktestBet, actual_total: float) -> tuple:
        """Grade total bet"""
        if actual_total == bet.line:
            return BetResult.PUSH, 0
        
        over = actual_total > bet.line
        bet_on_over = bet.predicted_side == 'over'
        
        if (over and bet_on_over) or (not over and not bet_on_over):
            profit = self._calculate_profit(bet.stake, bet.odds)
            return BetResult.WIN, profit
        else:
            return BetResult.LOSS, -bet.stake
    
    def _calculate_profit(self, stake: float, odds: int) -> float:
        """Calculate profit from winning bet"""
        if odds > 0:
            return stake * (odds / 100)
        else:
            return stake * (100 / abs(odds))
    
    def _calculate_clv(self, bet: BacktestBet, closing_data: Dict) -> float:
        """Calculate closing line value"""
        closing_line = closing_data.get('line', bet.line)
        
        if bet.bet_type == BetType.SPREAD:
            if bet.predicted_side == 'home':
                return closing_line - bet.line
            else:
                return bet.line - closing_line
        elif bet.bet_type == BetType.TOTAL:
            if bet.predicted_side == 'over':
                return bet.line - closing_line
            else:
                return closing_line - bet.line
        
        return 0
    
    def _check_stop_conditions(self, config: BacktestConfig, balance: float) -> bool:
        """Check if stop loss or take profit reached"""
        if config.stop_loss_percent:
            loss_threshold = config.initial_bankroll * (1 - config.stop_loss_percent)
            if balance <= loss_threshold:
                return True
        
        if config.take_profit_percent:
            profit_threshold = config.initial_bankroll * (1 + config.take_profit_percent)
            if balance >= profit_threshold:
                return True
        
        return False
    
    def _create_daily_snapshot(
        self,
        date: datetime,
        start_balance: float,
        end_balance: float,
        bets_placed: int,
        settled_bets: List[BacktestBet]
    ) -> DailySnapshot:
        """Create daily snapshot"""
        won = sum(1 for b in settled_bets if b.result == BetResult.WIN)
        lost = sum(1 for b in settled_bets if b.result == BetResult.LOSS)
        pushed = sum(1 for b in settled_bets if b.result == BetResult.PUSH)
        
        total_wagered = sum(b.stake for b in settled_bets)
        total_pl = sum(b.profit_loss for b in settled_bets)
        
        roi = (total_pl / total_wagered * 100) if total_wagered > 0 else 0
        win_rate = (won / (won + lost) * 100) if (won + lost) > 0 else 0
        avg_odds = sum(b.odds for b in settled_bets) / len(settled_bets) if settled_bets else 0
        avg_edge = sum(b.edge for b in settled_bets) / len(settled_bets) if settled_bets else 0
        
        return DailySnapshot(
            date=date,
            starting_balance=start_balance,
            ending_balance=end_balance,
            bets_placed=bets_placed,
            bets_won=won,
            bets_lost=lost,
            bets_pushed=pushed,
            total_wagered=total_wagered,
            total_profit_loss=total_pl,
            roi_percent=roi,
            win_rate=win_rate,
            avg_odds=avg_odds,
            avg_edge=avg_edge
        )
    
    def _calculate_results(
        self,
        config: BacktestConfig,
        run_id: str,
        started_at: datetime,
        completed_at: datetime,
        daily_snapshots: List[DailySnapshot],
        equity_curve: List[float],
        drawdown_curve: List[float]
    ) -> BacktestResult:
        """Calculate comprehensive backtest results"""
        
        all_bets = self._settled_bets
        
        # Basic counts
        total_bets = len(all_bets)
        bets_won = sum(1 for b in all_bets if b.result == BetResult.WIN)
        bets_lost = sum(1 for b in all_bets if b.result == BetResult.LOSS)
        bets_pushed = sum(1 for b in all_bets if b.result == BetResult.PUSH)
        
        # Profit metrics
        total_wagered = sum(b.stake for b in all_bets)
        total_profit_loss = sum(b.profit_loss for b in all_bets)
        total_roi = (total_profit_loss / total_wagered * 100) if total_wagered > 0 else 0
        
        win_rate = (bets_won / (bets_won + bets_lost) * 100) if (bets_won + bets_lost) > 0 else 0
        avg_bet_size = total_wagered / total_bets if total_bets > 0 else 0
        avg_odds = sum(b.odds for b in all_bets) / total_bets if total_bets > 0 else 0
        avg_edge = sum(b.edge for b in all_bets) / total_bets if total_bets > 0 else 0
        
        # Profit factor
        gross_profit = sum(b.profit_loss for b in all_bets if b.profit_loss > 0)
        gross_loss = abs(sum(b.profit_loss for b in all_bets if b.profit_loss < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Expected value
        avg_win = gross_profit / bets_won if bets_won > 0 else 0
        avg_loss = gross_loss / bets_lost if bets_lost > 0 else 0
        expected_value = (win_rate / 100 * avg_win) - ((1 - win_rate / 100) * avg_loss)
        
        # Risk metrics
        max_drawdown_pct = max(drawdown_curve) * 100 if drawdown_curve else 0
        max_drawdown_amt = config.initial_bankroll - self._trough_bankroll
        max_runup_pct = ((self._peak_bankroll - config.initial_bankroll) / config.initial_bankroll * 100) if config.initial_bankroll > 0 else 0
        
        # Sharpe ratio (simplified daily)
        import statistics
        if len(self._daily_returns) > 1:
            avg_return = statistics.mean(self._daily_returns)
            std_return = statistics.stdev(self._daily_returns)
            sharpe_ratio = (avg_return / std_return) * (252 ** 0.5) if std_return > 0 else 0
            
            # Sortino ratio (downside deviation)
            downside_returns = [r for r in self._daily_returns if r < 0]
            if downside_returns:
                downside_std = statistics.stdev(downside_returns)
                sortino_ratio = (avg_return / downside_std) * (252 ** 0.5) if downside_std > 0 else 0
            else:
                sortino_ratio = float('inf')
        else:
            sharpe_ratio = 0
            sortino_ratio = 0
        
        # Calmar ratio
        calmar_ratio = (total_roi / max_drawdown_pct) if max_drawdown_pct > 0 else float('inf')
        
        # CLV metrics
        clv_values = [b.clv for b in all_bets if b.clv is not None]
        avg_clv = sum(clv_values) / len(clv_values) if clv_values else 0
        clv_positive_rate = (sum(1 for c in clv_values if c > 0) / len(clv_values) * 100) if clv_values else 0
        
        # Breakdowns
        by_sport = self._calculate_breakdown(all_bets, 'sport')
        by_bet_type = self._calculate_breakdown(all_bets, 'bet_type')
        by_tier = self._calculate_breakdown(all_bets, 'signal_tier')
        by_month = self._calculate_monthly_breakdown(all_bets)
        
        return BacktestResult(
            config=config,
            run_id=run_id,
            started_at=started_at,
            completed_at=completed_at,
            initial_bankroll=config.initial_bankroll,
            final_bankroll=self._current_bankroll,
            total_profit_loss=total_profit_loss,
            total_roi_percent=total_roi,
            total_bets=total_bets,
            bets_won=bets_won,
            bets_lost=bets_lost,
            bets_pushed=bets_pushed,
            win_rate=win_rate,
            max_drawdown_percent=max_drawdown_pct,
            max_drawdown_amount=max_drawdown_amt,
            max_runup_percent=max_runup_pct,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            total_wagered=total_wagered,
            avg_bet_size=avg_bet_size,
            avg_odds=int(avg_odds),
            avg_edge=avg_edge,
            profit_factor=profit_factor,
            expected_value=expected_value,
            avg_clv=avg_clv,
            clv_positive_rate=clv_positive_rate,
            by_sport=by_sport,
            by_bet_type=by_bet_type,
            by_tier=by_tier,
            by_month=by_month,
            daily_snapshots=daily_snapshots,
            equity_curve=equity_curve,
            drawdown_curve=drawdown_curve,
            bets=all_bets
        )
    
    def _calculate_breakdown(self, bets: List[BacktestBet], field: str) -> Dict[str, Dict]:
        """Calculate breakdown by field"""
        breakdown = {}
        
        for bet in bets:
            key = getattr(bet, field)
            if isinstance(key, Enum):
                key = key.value
            
            if key not in breakdown:
                breakdown[key] = {
                    'total_bets': 0, 'won': 0, 'lost': 0, 'pushed': 0,
                    'wagered': 0, 'profit_loss': 0
                }
            
            breakdown[key]['total_bets'] += 1
            breakdown[key]['wagered'] += bet.stake
            breakdown[key]['profit_loss'] += bet.profit_loss
            
            if bet.result == BetResult.WIN:
                breakdown[key]['won'] += 1
            elif bet.result == BetResult.LOSS:
                breakdown[key]['lost'] += 1
            else:
                breakdown[key]['pushed'] += 1
        
        # Calculate rates
        for key in breakdown:
            data = breakdown[key]
            decided = data['won'] + data['lost']
            data['win_rate'] = (data['won'] / decided * 100) if decided > 0 else 0
            data['roi'] = (data['profit_loss'] / data['wagered'] * 100) if data['wagered'] > 0 else 0
        
        return breakdown
    
    def _calculate_monthly_breakdown(self, bets: List[BacktestBet]) -> Dict[str, Dict]:
        """Calculate monthly breakdown"""
        by_month = {}
        
        for bet in bets:
            month_key = bet.date.strftime('%Y-%m')
            
            if month_key not in by_month:
                by_month[month_key] = {
                    'total_bets': 0, 'won': 0, 'lost': 0,
                    'wagered': 0, 'profit_loss': 0
                }
            
            by_month[month_key]['total_bets'] += 1
            by_month[month_key]['wagered'] += bet.stake
            by_month[month_key]['profit_loss'] += bet.profit_loss
            
            if bet.result == BetResult.WIN:
                by_month[month_key]['won'] += 1
            elif bet.result == BetResult.LOSS:
                by_month[month_key]['lost'] += 1
        
        # Calculate rates
        for month in by_month:
            data = by_month[month]
            decided = data['won'] + data['lost']
            data['win_rate'] = (data['won'] / decided * 100) if decided > 0 else 0
            data['roi'] = (data['profit_loss'] / data['wagered'] * 100) if data['wagered'] > 0 else 0
        
        return by_month


def create_backtest_engine(db_session) -> BacktestEngine:
    """Factory function to create backtest engine with database sources"""
    
    def prediction_source(start_date: datetime, end_date: datetime) -> List[Dict]:
        """Load predictions from database"""
        # This would query the predictions table
        # Placeholder implementation
        return []
    
    def odds_source(game_id: str) -> Dict:
        """Load odds from database"""
        # This would query the odds table
        return {}
    
    def results_source(game_id: str) -> Dict:
        """Load game results from database"""
        # This would query the games table
        return {}
    
    def closing_line_source(game_id: str) -> Dict:
        """Load closing lines from database"""
        # This would query the closing_lines table
        return {}
    
    return BacktestEngine(
        prediction_source=prediction_source,
        odds_source=odds_source,
        results_source=results_source,
        closing_line_source=closing_line_source
    )
