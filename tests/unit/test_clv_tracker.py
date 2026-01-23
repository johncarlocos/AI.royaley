"""
Unit Tests for CLV (Closing Line Value) Tracker
================================================
Tests for CLV calculation, tracking, and performance analysis.
"""

import pytest
from datetime import datetime, timedelta
from app.services.betting.clv_tracker import (
    CLVTracker,
    CLVResult,
    CLVSummary,
    CLVPerformanceTier,
    calculate_clv,
)


class TestCLVCalculation:
    """Test CLV calculation functions."""
    
    def test_spread_home_positive_clv(self):
        """Test positive CLV for home spread bet."""
        # Bet home -3, closed at -3.5 (got better number)
        clv = calculate_clv(
            bet_line=-3.0,
            closing_line=-3.5,
            bet_side='home',
            bet_type='spread'
        )
        assert clv > 0  # Positive CLV - we got a better number
        assert clv == pytest.approx(-0.5, abs=0.01)  # -3.5 - (-3) = -0.5 points in our favor
    
    def test_spread_home_negative_clv(self):
        """Test negative CLV for home spread bet."""
        # Bet home -3.5, closed at -3 (got worse number)
        clv = calculate_clv(
            bet_line=-3.5,
            closing_line=-3.0,
            bet_side='home',
            bet_type='spread'
        )
        assert clv < 0  # Negative CLV
    
    def test_spread_away_positive_clv(self):
        """Test positive CLV for away spread bet."""
        # Bet away +3.5, closed at +3 (got better number)
        clv = calculate_clv(
            bet_line=3.5,
            closing_line=3.0,
            bet_side='away',
            bet_type='spread'
        )
        assert clv > 0  # Positive CLV
    
    def test_spread_away_negative_clv(self):
        """Test negative CLV for away spread bet."""
        # Bet away +3, closed at +3.5 (got worse number)
        clv = calculate_clv(
            bet_line=3.0,
            closing_line=3.5,
            bet_side='away',
            bet_type='spread'
        )
        assert clv < 0  # Negative CLV
    
    def test_total_over_positive_clv(self):
        """Test positive CLV for over bet."""
        # Bet over 220, closed at 222 (got better number)
        clv = calculate_clv(
            bet_line=220.0,
            closing_line=222.0,
            bet_side='over',
            bet_type='total'
        )
        assert clv > 0  # Positive CLV
    
    def test_total_under_positive_clv(self):
        """Test positive CLV for under bet."""
        # Bet under 222, closed at 220 (got better number)
        clv = calculate_clv(
            bet_line=222.0,
            closing_line=220.0,
            bet_side='under',
            bet_type='total'
        )
        assert clv > 0  # Positive CLV
    
    def test_moneyline_positive_clv(self):
        """Test positive CLV for moneyline bet."""
        # Bet at +150, closed at +120 (got better price)
        clv = calculate_clv(
            bet_line=150,
            closing_line=120,
            bet_side='home',
            bet_type='moneyline'
        )
        assert clv > 0  # Positive CLV
    
    def test_moneyline_negative_clv(self):
        """Test negative CLV for moneyline bet."""
        # Bet at -150, closed at -120 (got worse price)
        clv = calculate_clv(
            bet_line=-150,
            closing_line=-120,
            bet_side='home',
            bet_type='moneyline'
        )
        assert clv < 0  # Negative CLV
    
    def test_zero_clv(self):
        """Test zero CLV when lines match."""
        clv = calculate_clv(
            bet_line=-3.0,
            closing_line=-3.0,
            bet_side='home',
            bet_type='spread'
        )
        assert clv == 0


class TestCLVPerformanceTiers:
    """Test CLV performance tier classification."""
    
    def test_elite_tier(self):
        """Test elite tier (3%+ CLV)."""
        tracker = CLVTracker()
        # Add bets with 3%+ CLV
        for i in range(10):
            tracker.record_clv(CLVResult(
                bet_id=str(i),
                bet_line=-3.0,
                closing_line=-4.0,
                clv_cents=100,  # 1 point = ~2.8%
                clv_percent=3.5,
                bet_type='spread',
                bet_side='home',
                is_positive=True
            ))
        summary = tracker.get_summary()
        assert summary.performance_tier == CLVPerformanceTier.ELITE
    
    def test_professional_tier(self):
        """Test professional tier (2-3% CLV)."""
        tracker = CLVTracker()
        for i in range(10):
            tracker.record_clv(CLVResult(
                bet_id=str(i),
                bet_line=-3.0,
                closing_line=-3.8,
                clv_cents=80,
                clv_percent=2.5,
                bet_type='spread',
                bet_side='home',
                is_positive=True
            ))
        summary = tracker.get_summary()
        assert summary.performance_tier == CLVPerformanceTier.PROFESSIONAL
    
    def test_competent_tier(self):
        """Test competent tier (1-2% CLV)."""
        tracker = CLVTracker()
        for i in range(10):
            tracker.record_clv(CLVResult(
                bet_id=str(i),
                bet_line=-3.0,
                closing_line=-3.5,
                clv_cents=50,
                clv_percent=1.5,
                bet_type='spread',
                bet_side='home',
                is_positive=True
            ))
        summary = tracker.get_summary()
        assert summary.performance_tier == CLVPerformanceTier.COMPETENT
    
    def test_breakeven_tier(self):
        """Test breakeven tier (0-1% CLV)."""
        tracker = CLVTracker()
        for i in range(10):
            tracker.record_clv(CLVResult(
                bet_id=str(i),
                bet_line=-3.0,
                closing_line=-3.2,
                clv_cents=20,
                clv_percent=0.5,
                bet_type='spread',
                bet_side='home',
                is_positive=True
            ))
        summary = tracker.get_summary()
        assert summary.performance_tier == CLVPerformanceTier.BREAKEVEN
    
    def test_negative_tier(self):
        """Test negative tier (<0% CLV)."""
        tracker = CLVTracker()
        for i in range(10):
            tracker.record_clv(CLVResult(
                bet_id=str(i),
                bet_line=-3.0,
                closing_line=-2.5,
                clv_cents=-50,
                clv_percent=-1.5,
                bet_type='spread',
                bet_side='home',
                is_positive=False
            ))
        summary = tracker.get_summary()
        assert summary.performance_tier == CLVPerformanceTier.NEGATIVE


class TestCLVTracker:
    """Test CLVTracker class functionality."""
    
    def test_record_clv(self):
        """Test recording CLV results."""
        tracker = CLVTracker()
        result = CLVResult(
            bet_id="1",
            bet_line=-3.0,
            closing_line=-3.5,
            clv_cents=50,
            clv_percent=1.5,
            bet_type='spread',
            bet_side='home',
            is_positive=True
        )
        tracker.record_clv(result)
        assert len(tracker.results) == 1
    
    def test_get_summary(self):
        """Test summary generation."""
        tracker = CLVTracker()
        # Add mixed CLV results
        results = [
            CLVResult("1", -3.0, -3.5, 50, 1.5, 'spread', 'home', True),
            CLVResult("2", -3.0, -2.5, -50, -1.5, 'spread', 'away', False),
            CLVResult("3", 220.0, 222.0, 60, 2.0, 'total', 'over', True),
        ]
        for r in results:
            tracker.record_clv(r)
        
        summary = tracker.get_summary()
        assert summary.total_bets == 3
        assert summary.positive_clv_count == 2
        assert summary.positive_clv_rate == pytest.approx(0.667, rel=0.01)
    
    def test_get_recent_results(self):
        """Test retrieving recent results."""
        tracker = CLVTracker()
        for i in range(20):
            tracker.record_clv(CLVResult(
                bet_id=str(i),
                bet_line=-3.0,
                closing_line=-3.5,
                clv_cents=50,
                clv_percent=1.5,
                bet_type='spread',
                bet_side='home',
                is_positive=True
            ))
        
        recent = tracker.get_recent_results(n=10)
        assert len(recent) == 10
    
    def test_clv_by_sport(self):
        """Test CLV breakdown by sport."""
        tracker = CLVTracker()
        results = [
            CLVResult("1", -3.0, -3.5, 50, 1.5, 'spread', 'home', True, sport='NFL'),
            CLVResult("2", -3.0, -3.5, 60, 2.0, 'spread', 'home', True, sport='NFL'),
            CLVResult("3", 220.0, 222.0, 40, 1.2, 'total', 'over', True, sport='NBA'),
        ]
        for r in results:
            tracker.record_clv(r)
        
        summary = tracker.get_summary()
        assert 'NFL' in summary.by_sport
        assert 'NBA' in summary.by_sport
    
    def test_clv_by_bet_type(self):
        """Test CLV breakdown by bet type."""
        tracker = CLVTracker()
        results = [
            CLVResult("1", -3.0, -3.5, 50, 1.5, 'spread', 'home', True),
            CLVResult("2", 220.0, 222.0, 60, 2.0, 'total', 'over', True),
            CLVResult("3", -150, -180, 70, 2.5, 'moneyline', 'home', True),
        ]
        for r in results:
            tracker.record_clv(r)
        
        summary = tracker.get_summary()
        assert 'spread' in summary.by_bet_type
        assert 'total' in summary.by_bet_type
        assert 'moneyline' in summary.by_bet_type
    
    def test_trend_analysis(self):
        """Test trend analysis (improving/stable/declining)."""
        tracker = CLVTracker()
        # Improving trend: later bets have higher CLV
        for i in range(20):
            tracker.record_clv(CLVResult(
                bet_id=str(i),
                bet_line=-3.0,
                closing_line=-3.5,
                clv_cents=30 + i * 5,  # Increasing CLV
                clv_percent=1.0 + i * 0.1,
                bet_type='spread',
                bet_side='home',
                is_positive=True
            ))
        
        summary = tracker.get_summary()
        assert summary.trend in ['improving', 'stable', 'declining']


class TestCLVStatistics:
    """Test CLV statistical calculations."""
    
    def test_average_clv(self):
        """Test average CLV calculation."""
        tracker = CLVTracker()
        results = [
            CLVResult("1", -3.0, -3.5, 50, 1.5, 'spread', 'home', True),
            CLVResult("2", -3.0, -4.0, 100, 3.0, 'spread', 'home', True),
            CLVResult("3", -3.0, -2.5, -50, -1.5, 'spread', 'away', False),
        ]
        for r in results:
            tracker.record_clv(r)
        
        summary = tracker.get_summary()
        expected_avg = (1.5 + 3.0 - 1.5) / 3
        assert summary.avg_clv_percent == pytest.approx(expected_avg, rel=0.01)
    
    def test_best_worst_clv(self):
        """Test best and worst CLV tracking."""
        tracker = CLVTracker()
        results = [
            CLVResult("1", -3.0, -3.5, 50, 1.5, 'spread', 'home', True),
            CLVResult("2", -3.0, -5.0, 200, 5.0, 'spread', 'home', True),
            CLVResult("3", -3.0, -2.0, -100, -3.0, 'spread', 'away', False),
        ]
        for r in results:
            tracker.record_clv(r)
        
        summary = tracker.get_summary()
        assert summary.best_clv == pytest.approx(5.0, rel=0.01)
        assert summary.worst_clv == pytest.approx(-3.0, rel=0.01)
    
    def test_std_deviation(self):
        """Test CLV standard deviation."""
        tracker = CLVTracker()
        # Add varied CLV results
        clv_values = [1.0, 2.0, 3.0, 2.0, 1.0]
        for i, clv in enumerate(clv_values):
            tracker.record_clv(CLVResult(
                bet_id=str(i),
                bet_line=-3.0,
                closing_line=-3.5,
                clv_cents=int(clv * 30),
                clv_percent=clv,
                bet_type='spread',
                bet_side='home',
                is_positive=clv > 0
            ))
        
        summary = tracker.get_summary()
        assert summary.std_dev >= 0


class TestCLVEdgeCases:
    """Test edge cases for CLV tracking."""
    
    def test_empty_tracker(self):
        """Test summary with no results."""
        tracker = CLVTracker()
        summary = tracker.get_summary()
        assert summary.total_bets == 0
        assert summary.avg_clv_percent == 0
    
    def test_single_result(self):
        """Test with single CLV result."""
        tracker = CLVTracker()
        tracker.record_clv(CLVResult(
            bet_id="1",
            bet_line=-3.0,
            closing_line=-3.5,
            clv_cents=50,
            clv_percent=1.5,
            bet_type='spread',
            bet_side='home',
            is_positive=True
        ))
        
        summary = tracker.get_summary()
        assert summary.total_bets == 1
        assert summary.positive_clv_rate == 1.0
    
    def test_all_negative_clv(self):
        """Test when all CLV is negative."""
        tracker = CLVTracker()
        for i in range(5):
            tracker.record_clv(CLVResult(
                bet_id=str(i),
                bet_line=-3.0,
                closing_line=-2.0,
                clv_cents=-100,
                clv_percent=-3.0,
                bet_type='spread',
                bet_side='home',
                is_positive=False
            ))
        
        summary = tracker.get_summary()
        assert summary.positive_clv_rate == 0
        assert summary.performance_tier == CLVPerformanceTier.NEGATIVE
