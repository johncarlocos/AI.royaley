"""
Unit Tests for Kelly Criterion Calculator
==========================================
Tests for bet sizing, edge calculation, and bankroll management.
"""

import pytest
from decimal import Decimal
from app.services.betting.kelly_calculator import (
    KellyCalculator,
    KellyMode,
    BetSizing,
    calculate_kelly_bet,
    american_to_decimal,
    decimal_to_american,
    implied_probability,
    calculate_edge,
)


class TestOddsConversion:
    """Test odds conversion functions."""
    
    def test_american_to_decimal_positive(self):
        """Test positive American odds conversion."""
        assert american_to_decimal(100) == 2.0
        assert american_to_decimal(150) == 2.5
        assert american_to_decimal(200) == 3.0
        assert american_to_decimal(300) == 4.0
    
    def test_american_to_decimal_negative(self):
        """Test negative American odds conversion."""
        assert american_to_decimal(-100) == 2.0
        assert american_to_decimal(-110) == pytest.approx(1.909, rel=0.01)
        assert american_to_decimal(-150) == pytest.approx(1.667, rel=0.01)
        assert american_to_decimal(-200) == 1.5
    
    def test_decimal_to_american_favorites(self):
        """Test decimal to American for favorites."""
        assert decimal_to_american(1.5) == -200
        assert decimal_to_american(1.8) == pytest.approx(-125, abs=1)
        assert decimal_to_american(2.0) == 100
    
    def test_decimal_to_american_underdogs(self):
        """Test decimal to American for underdogs."""
        assert decimal_to_american(2.5) == 150
        assert decimal_to_american(3.0) == 200
        assert decimal_to_american(4.0) == 300
    
    def test_implied_probability(self):
        """Test implied probability calculation."""
        assert implied_probability(-110) == pytest.approx(0.524, rel=0.01)
        assert implied_probability(100) == 0.5
        assert implied_probability(-200) == pytest.approx(0.667, rel=0.01)
        assert implied_probability(200) == pytest.approx(0.333, rel=0.01)


class TestEdgeCalculation:
    """Test edge calculation."""
    
    def test_positive_edge(self):
        """Test positive edge scenarios."""
        # 55% probability at -110 odds
        edge = calculate_edge(0.55, -110)
        assert edge > 0
        assert edge == pytest.approx(0.0238, rel=0.01)
    
    def test_negative_edge(self):
        """Test negative edge (no value bet)."""
        # 45% probability at -110 odds
        edge = calculate_edge(0.45, -110)
        assert edge < 0
    
    def test_break_even_edge(self):
        """Test break-even edge."""
        # 52.4% probability at -110 odds
        edge = calculate_edge(0.524, -110)
        assert abs(edge) < 0.01


class TestKellyBetCalculation:
    """Test Kelly Criterion bet sizing."""
    
    def test_full_kelly(self):
        """Test full Kelly calculation."""
        result = calculate_kelly_bet(
            probability=0.60,
            american_odds=-110,
            bankroll=10000,
            kelly_mode=KellyMode.FULL
        )
        assert isinstance(result, BetSizing)
        assert result.recommended_bet > 0
        assert result.is_value_bet is True
        assert result.edge > 0
    
    def test_quarter_kelly(self):
        """Test quarter Kelly (25%) calculation."""
        full_result = calculate_kelly_bet(
            probability=0.60,
            american_odds=-110,
            bankroll=10000,
            kelly_mode=KellyMode.FULL
        )
        quarter_result = calculate_kelly_bet(
            probability=0.60,
            american_odds=-110,
            bankroll=10000,
            kelly_mode=KellyMode.QUARTER
        )
        assert quarter_result.recommended_bet == pytest.approx(
            full_result.recommended_bet * 0.25, rel=0.01
        )
    
    def test_half_kelly(self):
        """Test half Kelly (50%) calculation."""
        full_result = calculate_kelly_bet(
            probability=0.60,
            american_odds=-110,
            bankroll=10000,
            kelly_mode=KellyMode.FULL
        )
        half_result = calculate_kelly_bet(
            probability=0.60,
            american_odds=-110,
            bankroll=10000,
            kelly_mode=KellyMode.HALF
        )
        assert half_result.recommended_bet == pytest.approx(
            full_result.recommended_bet * 0.5, rel=0.01
        )
    
    def test_max_bet_cap(self):
        """Test maximum bet percentage cap."""
        result = calculate_kelly_bet(
            probability=0.90,  # Very high edge
            american_odds=100,
            bankroll=10000,
            kelly_mode=KellyMode.FULL,
            max_bet_percent=0.02  # 2% max
        )
        assert result.recommended_bet <= 10000 * 0.02
    
    def test_minimum_edge_threshold(self):
        """Test minimum edge threshold."""
        result = calculate_kelly_bet(
            probability=0.53,  # Slight edge
            american_odds=-110,
            bankroll=10000,
            kelly_mode=KellyMode.QUARTER,
            min_edge_threshold=0.05  # 5% minimum edge
        )
        # Edge is less than threshold, should be $0
        assert result.recommended_bet == 0
        assert result.is_value_bet is False
    
    def test_no_value_bet(self):
        """Test when probability doesn't justify bet."""
        result = calculate_kelly_bet(
            probability=0.45,
            american_odds=-110,
            bankroll=10000,
            kelly_mode=KellyMode.FULL
        )
        assert result.recommended_bet == 0
        assert result.is_value_bet is False
        assert result.edge < 0
    
    def test_fixed_bet_mode(self):
        """Test fixed bet sizing mode."""
        result = calculate_kelly_bet(
            probability=0.60,
            american_odds=-110,
            bankroll=10000,
            kelly_mode=KellyMode.FIXED,
            fixed_bet_amount=100
        )
        assert result.recommended_bet == 100
    
    def test_minimum_bet_amount(self):
        """Test minimum bet amount enforcement."""
        result = calculate_kelly_bet(
            probability=0.55,  # Small edge
            american_odds=-110,
            bankroll=1000,  # Small bankroll
            kelly_mode=KellyMode.QUARTER,
            min_bet_amount=10
        )
        if result.is_value_bet:
            assert result.recommended_bet >= 10 or result.recommended_bet == 0


class TestConfidenceTiers:
    """Test confidence tier classification."""
    
    def test_tier_a(self):
        """Test Tier A classification (65%+)."""
        result = calculate_kelly_bet(
            probability=0.68,
            american_odds=-110,
            bankroll=10000
        )
        assert result.confidence_tier == 'A'
    
    def test_tier_b(self):
        """Test Tier B classification (60-65%)."""
        result = calculate_kelly_bet(
            probability=0.62,
            american_odds=-110,
            bankroll=10000
        )
        assert result.confidence_tier == 'B'
    
    def test_tier_c(self):
        """Test Tier C classification (55-60%)."""
        result = calculate_kelly_bet(
            probability=0.57,
            american_odds=-110,
            bankroll=10000
        )
        assert result.confidence_tier == 'C'
    
    def test_tier_d(self):
        """Test Tier D classification (<55%)."""
        result = calculate_kelly_bet(
            probability=0.52,
            american_odds=-110,
            bankroll=10000
        )
        assert result.confidence_tier == 'D'


class TestRiskLevels:
    """Test risk level classification."""
    
    def test_low_risk(self):
        """Test low risk classification (≤0.5%)."""
        result = calculate_kelly_bet(
            probability=0.55,
            american_odds=-110,
            bankroll=10000,
            kelly_mode=KellyMode.EIGHTH
        )
        if result.bet_fraction <= 0.005:
            assert result.risk_level == 'low'
    
    def test_extreme_risk_capped(self):
        """Test extreme risk is capped."""
        result = calculate_kelly_bet(
            probability=0.85,
            american_odds=100,
            bankroll=10000,
            kelly_mode=KellyMode.FULL,
            max_bet_percent=0.02
        )
        # Even with high edge, max bet caps at 2%
        assert result.bet_fraction <= 0.02


class TestKellyCalculatorClass:
    """Test KellyCalculator class methods."""
    
    def test_calculator_initialization(self):
        """Test calculator initialization."""
        calc = KellyCalculator(
            kelly_mode=KellyMode.QUARTER,
            max_bet_percent=0.02,
            min_edge_threshold=0.03
        )
        assert calc.kelly_mode == KellyMode.QUARTER
        assert calc.max_bet_percent == 0.02
    
    def test_calculate_bet_size(self):
        """Test bet size calculation method."""
        calc = KellyCalculator()
        result = calc.calculate_bet_size(
            probability=0.60,
            american_odds=-110,
            bankroll=10000
        )
        assert isinstance(result, BetSizing)
    
    def test_multi_bet_sizing(self):
        """Test multiple bet sizing."""
        calc = KellyCalculator()
        bets = [
            {"probability": 0.60, "american_odds": -110},
            {"probability": 0.58, "american_odds": -105},
            {"probability": 0.62, "american_odds": 100},
        ]
        results = calc.calculate_multi_bet_sizing(
            bets=bets,
            bankroll=10000,
            max_daily_exposure=0.05
        )
        assert len(results) == 3
        total_exposure = sum(r.recommended_bet for r in results)
        assert total_exposure <= 10000 * 0.05
    
    def test_growth_rate_estimation(self):
        """Test expected growth rate calculation."""
        calc = KellyCalculator()
        growth_rate = calc.estimate_growth_rate(
            probability=0.60,
            american_odds=-110,
            kelly_fraction=0.25
        )
        assert growth_rate > 0  # Positive edge should have positive growth


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_probability_bounds(self):
        """Test probability at boundary values."""
        # Probability at 50%
        result = calculate_kelly_bet(
            probability=0.50,
            american_odds=100,
            bankroll=10000
        )
        assert result.recommended_bet == 0  # No edge
    
    def test_very_high_odds(self):
        """Test with very high odds (longshot)."""
        result = calculate_kelly_bet(
            probability=0.15,
            american_odds=800,
            bankroll=10000
        )
        # Small edge on longshot
        assert result.recommended_bet >= 0
    
    def test_heavy_favorite(self):
        """Test with heavy favorite odds."""
        result = calculate_kelly_bet(
            probability=0.85,
            american_odds=-300,
            bankroll=10000
        )
        assert result.recommended_bet >= 0
    
    def test_zero_bankroll(self):
        """Test with zero bankroll."""
        result = calculate_kelly_bet(
            probability=0.60,
            american_odds=-110,
            bankroll=0
        )
        assert result.recommended_bet == 0
    
    def test_negative_american_odds_conversion(self):
        """Test edge case with -100 odds."""
        decimal = american_to_decimal(-100)
        assert decimal == 2.0
