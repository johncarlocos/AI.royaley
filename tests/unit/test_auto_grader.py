"""
Unit Tests for Auto-Grader
===========================
Tests for automatic prediction grading and profit/loss calculation.
"""

import pytest
from datetime import datetime
from app.services.betting.auto_grader import (
    AutoGrader,
    GradeResult,
    GradingOutcome,
    GradingSummary,
    grade_spread_bet,
    grade_moneyline_bet,
    grade_total_bet,
    calculate_profit_loss,
)


class TestSpreadGrading:
    """Test spread bet grading."""
    
    def test_home_cover_win(self):
        """Test home team covering spread - win."""
        result = grade_spread_bet(
            home_score=28,
            away_score=21,
            spread=-3.5,
            predicted_side='home'
        )
        assert result == GradeResult.WIN
        # Home won by 7, spread was -3.5, covered by 3.5
    
    def test_home_cover_loss(self):
        """Test home team not covering spread - loss."""
        result = grade_spread_bet(
            home_score=24,
            away_score=21,
            spread=-7.0,
            predicted_side='home'
        )
        assert result == GradeResult.LOSS
        # Home won by 3, needed to win by 7+
    
    def test_away_cover_win(self):
        """Test away team covering spread - win."""
        result = grade_spread_bet(
            home_score=21,
            away_score=28,
            spread=-3.5,
            predicted_side='away'
        )
        assert result == GradeResult.WIN
        # Away won by 7, getting +3.5
    
    def test_away_cover_loss(self):
        """Test away team not covering spread - loss."""
        result = grade_spread_bet(
            home_score=35,
            away_score=21,
            spread=-3.5,
            predicted_side='away'
        )
        assert result == GradeResult.LOSS
        # Away lost by 14, even with +3.5
    
    def test_push_on_spread(self):
        """Test push when result lands on spread."""
        result = grade_spread_bet(
            home_score=24,
            away_score=21,
            spread=-3.0,
            predicted_side='home'
        )
        assert result == GradeResult.PUSH
        # Home won by exactly 3 with -3 spread
    
    def test_home_favorite_loss(self):
        """Test home favorite losing outright."""
        result = grade_spread_bet(
            home_score=20,
            away_score=24,
            spread=-3.5,
            predicted_side='home'
        )
        assert result == GradeResult.LOSS
    
    def test_home_underdog_win(self):
        """Test home underdog covering."""
        result = grade_spread_bet(
            home_score=21,
            away_score=24,
            spread=7.0,  # Home is +7 underdog
            predicted_side='home'
        )
        assert result == GradeResult.WIN
        # Home lost by 3, but had +7


class TestMoneylineGrading:
    """Test moneyline bet grading."""
    
    def test_home_win(self):
        """Test home team winning moneyline."""
        result = grade_moneyline_bet(
            home_score=28,
            away_score=21,
            predicted_side='home'
        )
        assert result == GradeResult.WIN
    
    def test_home_loss(self):
        """Test home team losing moneyline."""
        result = grade_moneyline_bet(
            home_score=21,
            away_score=28,
            predicted_side='home'
        )
        assert result == GradeResult.LOSS
    
    def test_away_win(self):
        """Test away team winning moneyline."""
        result = grade_moneyline_bet(
            home_score=21,
            away_score=28,
            predicted_side='away'
        )
        assert result == GradeResult.WIN
    
    def test_away_loss(self):
        """Test away team losing moneyline."""
        result = grade_moneyline_bet(
            home_score=28,
            away_score=21,
            predicted_side='away'
        )
        assert result == GradeResult.LOSS
    
    def test_tie_push(self):
        """Test tie results in push."""
        result = grade_moneyline_bet(
            home_score=21,
            away_score=21,
            predicted_side='home'
        )
        assert result == GradeResult.PUSH


class TestTotalGrading:
    """Test total (over/under) bet grading."""
    
    def test_over_win(self):
        """Test over bet winning."""
        result = grade_total_bet(
            home_score=28,
            away_score=24,
            total_line=45.5,
            predicted_side='over'
        )
        assert result == GradeResult.WIN
        # Total is 52, over 45.5
    
    def test_over_loss(self):
        """Test over bet losing."""
        result = grade_total_bet(
            home_score=17,
            away_score=14,
            total_line=45.5,
            predicted_side='over'
        )
        assert result == GradeResult.LOSS
        # Total is 31, under 45.5
    
    def test_under_win(self):
        """Test under bet winning."""
        result = grade_total_bet(
            home_score=17,
            away_score=14,
            total_line=45.5,
            predicted_side='under'
        )
        assert result == GradeResult.WIN
        # Total is 31, under 45.5
    
    def test_under_loss(self):
        """Test under bet losing."""
        result = grade_total_bet(
            home_score=28,
            away_score=24,
            total_line=45.5,
            predicted_side='under'
        )
        assert result == GradeResult.LOSS
        # Total is 52, over 45.5
    
    def test_total_push(self):
        """Test push when total lands on line."""
        result = grade_total_bet(
            home_score=24,
            away_score=21,
            total_line=45.0,
            predicted_side='over'
        )
        assert result == GradeResult.PUSH
        # Total is exactly 45


class TestProfitLossCalculation:
    """Test profit/loss calculation."""
    
    def test_win_positive_odds(self):
        """Test profit on win with positive odds."""
        profit = calculate_profit_loss(
            result=GradeResult.WIN,
            stake=100,
            american_odds=150
        )
        assert profit == 150  # Win $150 on $100 bet at +150
    
    def test_win_negative_odds(self):
        """Test profit on win with negative odds."""
        profit = calculate_profit_loss(
            result=GradeResult.WIN,
            stake=110,
            american_odds=-110
        )
        assert profit == 100  # Win $100 on $110 bet at -110
    
    def test_loss(self):
        """Test loss calculation."""
        profit = calculate_profit_loss(
            result=GradeResult.LOSS,
            stake=100,
            american_odds=-110
        )
        assert profit == -100  # Lose entire stake
    
    def test_push(self):
        """Test push returns zero."""
        profit = calculate_profit_loss(
            result=GradeResult.PUSH,
            stake=100,
            american_odds=-110
        )
        assert profit == 0  # No profit/loss on push
    
    def test_win_even_money(self):
        """Test win at even money (+100)."""
        profit = calculate_profit_loss(
            result=GradeResult.WIN,
            stake=100,
            american_odds=100
        )
        assert profit == 100
    
    def test_win_heavy_favorite(self):
        """Test win on heavy favorite (-300)."""
        profit = calculate_profit_loss(
            result=GradeResult.WIN,
            stake=300,
            american_odds=-300
        )
        assert profit == 100  # Win $100 on $300 bet at -300
    
    def test_win_big_underdog(self):
        """Test win on big underdog (+300)."""
        profit = calculate_profit_loss(
            result=GradeResult.WIN,
            stake=100,
            american_odds=300
        )
        assert profit == 300  # Win $300 on $100 bet at +300


class TestAutoGraderClass:
    """Test AutoGrader class functionality."""
    
    def test_grade_spread_prediction(self):
        """Test grading a spread prediction."""
        grader = AutoGrader()
        outcome = grader.grade_prediction(
            prediction_id="pred_001",
            bet_type='spread',
            predicted_side='home',
            line=-3.5,
            odds=-110,
            stake=100,
            home_score=28,
            away_score=21
        )
        assert isinstance(outcome, GradingOutcome)
        assert outcome.result == GradeResult.WIN
        assert outcome.profit_loss > 0
    
    def test_grade_moneyline_prediction(self):
        """Test grading a moneyline prediction."""
        grader = AutoGrader()
        outcome = grader.grade_prediction(
            prediction_id="pred_002",
            bet_type='moneyline',
            predicted_side='home',
            line=0,  # Not applicable for ML
            odds=-150,
            stake=150,
            home_score=28,
            away_score=21
        )
        assert outcome.result == GradeResult.WIN
        assert outcome.profit_loss == 100  # Win $100 on $150 at -150
    
    def test_grade_total_prediction(self):
        """Test grading a total prediction."""
        grader = AutoGrader()
        outcome = grader.grade_prediction(
            prediction_id="pred_003",
            bet_type='total',
            predicted_side='over',
            line=45.5,
            odds=-110,
            stake=110,
            home_score=28,
            away_score=24
        )
        assert outcome.result == GradeResult.WIN
    
    def test_get_summary(self):
        """Test summary generation."""
        grader = AutoGrader()
        
        # Grade several predictions
        predictions = [
            {"id": "1", "bet_type": "spread", "side": "home", "line": -3.5, 
             "odds": -110, "stake": 100, "home": 28, "away": 21},  # Win
            {"id": "2", "bet_type": "spread", "side": "home", "line": -7.0,
             "odds": -110, "stake": 100, "home": 24, "away": 21},  # Loss
            {"id": "3", "bet_type": "moneyline", "side": "home", "line": 0,
             "odds": 100, "stake": 100, "home": 28, "away": 21},  # Win
        ]
        
        for p in predictions:
            grader.grade_prediction(
                prediction_id=p["id"],
                bet_type=p["bet_type"],
                predicted_side=p["side"],
                line=p["line"],
                odds=p["odds"],
                stake=p["stake"],
                home_score=p["home"],
                away_score=p["away"]
            )
        
        summary = grader.get_summary()
        assert summary.total_graded == 3
        assert summary.wins == 2
        assert summary.losses == 1
        assert summary.win_rate == pytest.approx(0.667, rel=0.01)
    
    def test_get_recent_grades(self):
        """Test retrieving recent grades."""
        grader = AutoGrader()
        for i in range(10):
            grader.grade_prediction(
                prediction_id=str(i),
                bet_type='moneyline',
                predicted_side='home',
                line=0,
                odds=-110,
                stake=100,
                home_score=28,
                away_score=21
            )
        
        recent = grader.get_recent_grades(n=5)
        assert len(recent) == 5


class TestGradingSummary:
    """Test grading summary calculations."""
    
    def test_win_rate_calculation(self):
        """Test accurate win rate calculation."""
        grader = AutoGrader()
        
        # 6 wins, 4 losses = 60% win rate
        for i in range(6):
            grader.grade_prediction(
                prediction_id=f"w{i}",
                bet_type='moneyline',
                predicted_side='home',
                line=0, odds=-110, stake=100,
                home_score=28, away_score=21
            )
        for i in range(4):
            grader.grade_prediction(
                prediction_id=f"l{i}",
                bet_type='moneyline',
                predicted_side='home',
                line=0, odds=-110, stake=100,
                home_score=21, away_score=28
            )
        
        summary = grader.get_summary()
        assert summary.win_rate == 0.60
    
    def test_roi_calculation(self):
        """Test ROI calculation."""
        grader = AutoGrader()
        
        # Win at +100: profit $100 on $100
        grader.grade_prediction(
            prediction_id="1",
            bet_type='moneyline',
            predicted_side='home',
            line=0, odds=100, stake=100,
            home_score=28, away_score=21
        )
        # Loss: -$100
        grader.grade_prediction(
            prediction_id="2",
            bet_type='moneyline',
            predicted_side='home',
            line=0, odds=100, stake=100,
            home_score=21, away_score=28
        )
        
        summary = grader.get_summary()
        # Total wagered: $200, Net P/L: $0, ROI: 0%
        assert summary.total_wagered == 200
        assert summary.profit_loss == 0
        assert summary.roi == 0
    
    def test_pushes_excluded_from_win_rate(self):
        """Test that pushes are excluded from win rate."""
        grader = AutoGrader()
        
        # 1 win, 1 loss, 1 push
        grader.grade_prediction(
            prediction_id="1", bet_type='spread', predicted_side='home',
            line=-3.5, odds=-110, stake=100,
            home_score=28, away_score=21  # Win
        )
        grader.grade_prediction(
            prediction_id="2", bet_type='spread', predicted_side='home',
            line=-7.0, odds=-110, stake=100,
            home_score=24, away_score=21  # Loss
        )
        grader.grade_prediction(
            prediction_id="3", bet_type='spread', predicted_side='home',
            line=-3.0, odds=-110, stake=100,
            home_score=24, away_score=21  # Push (won by exactly 3)
        )
        
        summary = grader.get_summary()
        assert summary.pushes == 1
        # Win rate should be 1/2 = 50% (excluding push)
        assert summary.win_rate == 0.50


class TestEdgeCases:
    """Test edge cases for grading."""
    
    def test_overtime_game(self):
        """Test grading overtime game."""
        result = grade_moneyline_bet(
            home_score=31,  # Won in OT
            away_score=28,
            predicted_side='home'
        )
        assert result == GradeResult.WIN
    
    def test_high_scoring_game(self):
        """Test high scoring game total."""
        result = grade_total_bet(
            home_score=56,
            away_score=49,
            total_line=75.5,
            predicted_side='over'
        )
        assert result == GradeResult.WIN
        # Total is 105, well over 75.5
    
    def test_defensive_game(self):
        """Test low scoring game."""
        result = grade_total_bet(
            home_score=13,
            away_score=10,
            total_line=45.5,
            predicted_side='under'
        )
        assert result == GradeResult.WIN
        # Total is 23, well under 45.5
    
    def test_shutout_game(self):
        """Test shutout game."""
        result = grade_moneyline_bet(
            home_score=28,
            away_score=0,
            predicted_side='home'
        )
        assert result == GradeResult.WIN
    
    def test_large_spread(self):
        """Test large point spread."""
        result = grade_spread_bet(
            home_score=42,
            away_score=7,
            spread=-28.5,
            predicted_side='home'
        )
        assert result == GradeResult.WIN
        # Home won by 35, covered -28.5
