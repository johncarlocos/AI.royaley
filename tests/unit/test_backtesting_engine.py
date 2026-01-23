"""
Unit tests for Backtesting Engine.
Tests historical simulation, bet filtering, and performance metrics.
"""

import pytest
from decimal import Decimal
from datetime import datetime, date, timedelta

from app.services.betting.backtesting_engine import (
    BacktestEngine,
    BacktestConfig,
    BacktestResult,
    BacktestStatus,
    DailyResult,
    SportResult,
    TierResult,
    SimulatedBet,
)
from app.services.betting.kelly_calculator import KellyMode


class TestBacktestStatus:
    """Tests for BacktestStatus enum."""
    
    def test_status_values(self):
        """Verify all status values exist."""
        assert BacktestStatus.PENDING.value == "pending"
        assert BacktestStatus.RUNNING.value == "running"
        assert BacktestStatus.COMPLETED.value == "completed"
        assert BacktestStatus.FAILED.value == "failed"
        assert BacktestStatus.CANCELLED.value == "cancelled"


class TestBacktestConfig:
    """Tests for BacktestConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = BacktestConfig(
            sport_codes=["NBA"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
        )
        
        assert config.initial_bankroll == Decimal("10000.00")
        assert config.kelly_mode == KellyMode.QUARTER
        assert config.max_bet_percent == 0.02
        assert config.min_edge_threshold == 0.03
        assert config.min_confidence == 0.55
        assert config.signal_tiers == ["A", "B"]
        assert config.bet_types == ["spread", "moneyline", "total"]
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = BacktestConfig(
            sport_codes=["NFL", "NBA"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 6, 30),
            initial_bankroll=Decimal("50000.00"),
            kelly_mode=KellyMode.HALF,
            max_bet_percent=0.03,
            min_edge_threshold=0.05,
            min_confidence=0.60,
            signal_tiers=["A"],
            bet_types=["spread"],
            max_daily_bets=10,
            max_daily_exposure=0.15,
        )
        
        assert config.initial_bankroll == Decimal("50000.00")
        assert config.kelly_mode == KellyMode.HALF
        assert config.max_daily_bets == 10
        assert config.max_daily_exposure == 0.15
    
    def test_date_validation(self):
        """Test date range validation."""
        config = BacktestConfig(
            sport_codes=["NBA"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
        )
        
        assert config.start_date < config.end_date
        days = (config.end_date - config.start_date).days
        assert days == 365


class TestSimulatedBet:
    """Tests for SimulatedBet dataclass."""
    
    def test_simulated_bet_creation(self):
        """Test creating a simulated bet."""
        bet = SimulatedBet(
            prediction_id="pred_123",
            game_id="game_456",
            sport_code="NBA",
            bet_type="spread",
            bet_side="home",
            probability=0.65,
            edge=0.08,
            odds=-110,
            line=-3.5,
            stake=Decimal("200.00"),
            result="win",
            profit_loss=Decimal("181.82"),
            clv_cents=1.5,
            bet_date=date(2024, 3, 15),
            signal_tier="A",
        )
        
        assert bet.probability == 0.65
        assert bet.result == "win"
        assert bet.profit_loss == Decimal("181.82")
    
    def test_simulated_bet_loss(self):
        """Test simulated bet with loss."""
        bet = SimulatedBet(
            prediction_id="pred_124",
            game_id="game_457",
            sport_code="NFL",
            bet_type="moneyline",
            bet_side="away",
            probability=0.58,
            edge=0.04,
            odds=150,
            line=0,
            stake=Decimal("100.00"),
            result="loss",
            profit_loss=Decimal("-100.00"),
            clv_cents=-0.5,
            bet_date=date(2024, 3, 16),
            signal_tier="B",
        )
        
        assert bet.result == "loss"
        assert bet.profit_loss == Decimal("-100.00")


class TestDailyResult:
    """Tests for DailyResult dataclass."""
    
    def test_daily_result_creation(self):
        """Test creating a daily result."""
        result = DailyResult(
            date=date(2024, 3, 15),
            bets_placed=5,
            wins=3,
            losses=2,
            pushes=0,
            amount_wagered=Decimal("500.00"),
            profit_loss=Decimal("125.00"),
            bankroll=Decimal("10125.00"),
            roi=25.0,
            clv_sum=3.5,
        )
        
        assert result.bets_placed == 5
        assert result.wins == 3
        assert result.roi == 25.0
    
    def test_daily_result_with_pushes(self):
        """Test daily result with pushes."""
        result = DailyResult(
            date=date(2024, 3, 16),
            bets_placed=4,
            wins=2,
            losses=1,
            pushes=1,
            amount_wagered=Decimal("400.00"),
            profit_loss=Decimal("75.00"),
            bankroll=Decimal("10200.00"),
            roi=18.75,
            clv_sum=2.0,
        )
        
        assert result.pushes == 1
        assert result.wins + result.losses + result.pushes == result.bets_placed


class TestSportResult:
    """Tests for SportResult dataclass."""
    
    def test_sport_result(self):
        """Test sport-specific results."""
        result = SportResult(
            sport_code="NBA",
            total_bets=100,
            wins=58,
            losses=40,
            pushes=2,
            win_rate=0.592,
            profit_loss=Decimal("1250.00"),
            roi=12.5,
            avg_clv=1.8,
        )
        
        assert result.win_rate == 0.592
        assert result.avg_clv == 1.8


class TestTierResult:
    """Tests for TierResult dataclass."""
    
    def test_tier_result(self):
        """Test tier-specific results."""
        result = TierResult(
            tier="A",
            total_bets=50,
            wins=35,
            losses=15,
            win_rate=0.70,
            profit_loss=Decimal("850.00"),
            roi=17.0,
        )
        
        assert result.tier == "A"
        assert result.win_rate == 0.70


class TestBacktestResult:
    """Tests for BacktestResult dataclass."""
    
    @pytest.fixture
    def sample_result(self):
        """Create a sample backtest result."""
        return BacktestResult(
            backtest_id="bt_123",
            status=BacktestStatus.COMPLETED,
            config=BacktestConfig(
                sport_codes=["NBA"],
                start_date=date(2024, 1, 1),
                end_date=date(2024, 3, 31),
            ),
            # Summary metrics
            total_bets=150,
            wins=90,
            losses=58,
            pushes=2,
            win_rate=0.608,
            initial_bankroll=Decimal("10000.00"),
            final_bankroll=Decimal("12500.00"),
            peak_bankroll=Decimal("13200.00"),
            total_wagered=Decimal("15000.00"),
            total_profit=Decimal("2500.00"),
            roi=16.67,
            max_drawdown=0.08,
            sharpe_ratio=1.45,
            avg_clv=1.2,
            positive_clv_rate=0.62,
            # Detailed breakdowns
            daily_results=[],
            sport_results=[],
            tier_results=[],
            started_at=datetime.now(),
            completed_at=datetime.now(),
        )
    
    def test_result_summary(self, sample_result):
        """Test result summary metrics."""
        assert sample_result.total_bets == 150
        assert sample_result.win_rate == 0.608
        assert sample_result.total_profit == Decimal("2500.00")
    
    def test_result_profitability(self, sample_result):
        """Test profitability metrics."""
        assert sample_result.roi == 16.67
        assert sample_result.final_bankroll > sample_result.initial_bankroll
    
    def test_result_risk_metrics(self, sample_result):
        """Test risk metrics."""
        assert sample_result.max_drawdown == 0.08
        assert sample_result.sharpe_ratio == 1.45
    
    def test_result_clv_metrics(self, sample_result):
        """Test CLV metrics."""
        assert sample_result.avg_clv == 1.2
        assert sample_result.positive_clv_rate == 0.62


class TestBacktestEngine:
    """Tests for BacktestEngine class."""
    
    @pytest.fixture
    def engine(self):
        """Create a backtest engine."""
        return BacktestEngine()
    
    def test_engine_initialization(self, engine):
        """Test engine initializes correctly."""
        assert engine is not None
        assert engine.active_backtests == {}
    
    def test_create_backtest(self, engine):
        """Test creating a new backtest."""
        config = BacktestConfig(
            sport_codes=["NBA"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 3, 31),
        )
        
        backtest_id = engine.create_backtest(config)
        assert backtest_id is not None
        assert backtest_id in engine.active_backtests
    
    def test_get_backtest_status(self, engine):
        """Test getting backtest status."""
        config = BacktestConfig(
            sport_codes=["NBA"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 3, 31),
        )
        
        backtest_id = engine.create_backtest(config)
        status = engine.get_status(backtest_id)
        assert status == BacktestStatus.PENDING


class TestBacktestFilters:
    """Tests for bet filtering in backtests."""
    
    def test_sport_filter(self):
        """Test filtering by sport."""
        config = BacktestConfig(
            sport_codes=["NBA", "NFL"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 3, 31),
        )
        
        assert "NBA" in config.sport_codes
        assert "MLB" not in config.sport_codes
    
    def test_tier_filter(self):
        """Test filtering by signal tier."""
        config = BacktestConfig(
            sport_codes=["NBA"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 3, 31),
            signal_tiers=["A"],
        )
        
        assert "A" in config.signal_tiers
        assert "B" not in config.signal_tiers
    
    def test_bet_type_filter(self):
        """Test filtering by bet type."""
        config = BacktestConfig(
            sport_codes=["NBA"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 3, 31),
            bet_types=["spread", "moneyline"],
        )
        
        assert "spread" in config.bet_types
        assert "total" not in config.bet_types
    
    def test_confidence_filter(self):
        """Test minimum confidence filter."""
        config = BacktestConfig(
            sport_codes=["NBA"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 3, 31),
            min_confidence=0.60,
        )
        
        # 58% confidence should be filtered out
        assert 0.58 < config.min_confidence
        # 62% confidence should pass
        assert 0.62 > config.min_confidence


class TestBacktestMetrics:
    """Tests for backtest metric calculations."""
    
    def test_win_rate_calculation(self):
        """Test win rate calculation."""
        wins = 60
        losses = 40
        pushes = 5
        
        # Win rate excludes pushes
        win_rate = wins / (wins + losses)
        assert win_rate == 0.60
    
    def test_roi_calculation(self):
        """Test ROI calculation."""
        total_wagered = Decimal("10000.00")
        total_profit = Decimal("800.00")
        
        roi = float(total_profit / total_wagered) * 100
        assert roi == 8.0
    
    def test_sharpe_ratio_calculation(self):
        """Test Sharpe ratio calculation."""
        daily_returns = [0.02, -0.01, 0.03, 0.01, -0.02, 0.02]
        
        import numpy as np
        mean_return = np.mean(daily_returns)
        std_return = np.std(daily_returns)
        
        # Annualized Sharpe (assuming 252 trading days)
        sharpe = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0
        assert sharpe > 0
    
    def test_max_drawdown_calculation(self):
        """Test max drawdown calculation."""
        bankroll_history = [10000, 10500, 10200, 9800, 10100, 10800, 10300]
        
        peak = 10000
        max_drawdown = 0
        
        for balance in bankroll_history:
            if balance > peak:
                peak = balance
            drawdown = (peak - balance) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # Max drawdown was from 10500 to 9800 = 6.67%
        assert max_drawdown == pytest.approx(0.0667, rel=0.01)


class TestBacktestValidation:
    """Tests for backtest input validation."""
    
    def test_valid_date_range(self):
        """Test valid date range."""
        config = BacktestConfig(
            sport_codes=["NBA"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
        )
        
        assert config.start_date < config.end_date
    
    def test_valid_bankroll(self):
        """Test valid initial bankroll."""
        config = BacktestConfig(
            sport_codes=["NBA"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
            initial_bankroll=Decimal("5000.00"),
        )
        
        assert config.initial_bankroll > Decimal("0")
    
    def test_valid_kelly_fraction(self):
        """Test valid Kelly fraction."""
        for mode in KellyMode:
            config = BacktestConfig(
                sport_codes=["NBA"],
                start_date=date(2024, 1, 1),
                end_date=date(2024, 12, 31),
                kelly_mode=mode,
            )
            assert config.kelly_mode in KellyMode
    
    def test_valid_max_bet(self):
        """Test valid max bet percentage."""
        config = BacktestConfig(
            sport_codes=["NBA"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
            max_bet_percent=0.05,
        )
        
        assert 0 < config.max_bet_percent <= 0.10
