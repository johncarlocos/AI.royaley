"""
ROYALEY - Core Unit Tests
Comprehensive tests for core components
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import hashlib
import json

# Test configuration
pytestmark = pytest.mark.unit


class TestELORatings:
    """Test ELO rating calculations."""
    
    def test_expected_score_equal_ratings(self):
        """Equal ratings should give 0.5 expected score."""
        from app.services.ml.features import calculate_expected_score
        
        expected = calculate_expected_score(1500, 1500)
        assert abs(expected - 0.5) < 0.001
    
    def test_expected_score_higher_rating_favored(self):
        """Higher rated team should have expected score > 0.5."""
        from app.services.ml.features import calculate_expected_score
        
        expected = calculate_expected_score(1600, 1400)
        assert expected > 0.5
        assert expected < 1.0
    
    def test_expected_score_lower_rating_underdog(self):
        """Lower rated team should have expected score < 0.5."""
        from app.services.ml.features import calculate_expected_score
        
        expected = calculate_expected_score(1400, 1600)
        assert expected < 0.5
        assert expected > 0.0
    
    def test_elo_update_win(self):
        """Winner should gain ELO points."""
        from app.services.ml.features import update_elo
        
        new_rating = update_elo(1500, 0.5, 1.0, k_factor=32)
        assert new_rating > 1500
    
    def test_elo_update_loss(self):
        """Loser should lose ELO points."""
        from app.services.ml.features import update_elo
        
        new_rating = update_elo(1500, 0.5, 0.0, k_factor=32)
        assert new_rating < 1500
    
    def test_elo_update_upset_win(self):
        """Upset win should gain more points."""
        from app.services.ml.features import update_elo
        
        upset_gain = update_elo(1400, 0.3, 1.0, k_factor=32) - 1400
        expected_gain = update_elo(1600, 0.7, 1.0, k_factor=32) - 1600
        
        assert upset_gain > expected_gain


class TestKellyCriterion:
    """Test Kelly Criterion bet sizing."""
    
    def test_kelly_positive_edge(self):
        """Positive edge should return positive bet size."""
        from app.services.betting.kelly import calculate_kelly_bet
        
        bet_fraction = calculate_kelly_bet(
            probability=0.60,
            decimal_odds=2.0,
            kelly_fraction=0.25
        )
        
        assert bet_fraction > 0
    
    def test_kelly_no_edge(self):
        """No edge should return zero bet size."""
        from app.services.betting.kelly import calculate_kelly_bet
        
        bet_fraction = calculate_kelly_bet(
            probability=0.50,
            decimal_odds=2.0,
            kelly_fraction=0.25
        )
        
        assert bet_fraction <= 0
    
    def test_kelly_negative_edge(self):
        """Negative edge should return zero bet size."""
        from app.services.betting.kelly import calculate_kelly_bet
        
        bet_fraction = calculate_kelly_bet(
            probability=0.40,
            decimal_odds=2.0,
            kelly_fraction=0.25
        )
        
        assert bet_fraction <= 0
    
    def test_kelly_max_bet_cap(self):
        """Bet should be capped at max bet percentage."""
        from app.services.betting.kelly import calculate_kelly_bet
        
        bet_fraction = calculate_kelly_bet(
            probability=0.90,
            decimal_odds=3.0,
            kelly_fraction=0.25,
            max_bet=0.02
        )
        
        assert bet_fraction <= 0.02
    
    def test_kelly_fractional_reduces_size(self):
        """Fractional Kelly should reduce bet size."""
        from app.services.betting.kelly import calculate_kelly_bet
        
        full_kelly = calculate_kelly_bet(
            probability=0.65,
            decimal_odds=2.0,
            kelly_fraction=1.0,
            max_bet=1.0
        )
        
        quarter_kelly = calculate_kelly_bet(
            probability=0.65,
            decimal_odds=2.0,
            kelly_fraction=0.25,
            max_bet=1.0
        )
        
        assert abs(quarter_kelly - full_kelly * 0.25) < 0.001


class TestCLVCalculation:
    """Test Closing Line Value calculations."""
    
    def test_clv_positive_spread_home(self):
        """Betting home spread that moves in favor should be positive CLV."""
        from app.services.betting.clv import calculate_clv
        
        clv = calculate_clv(
            bet_line=-3.0,
            closing_line=-4.5,
            bet_side="home"
        )
        
        assert clv > 0  # Got -3, closed at -4.5, good for home bettor
    
    def test_clv_negative_spread_away(self):
        """Betting away spread that moves against should be negative CLV."""
        from app.services.betting.clv import calculate_clv
        
        clv = calculate_clv(
            bet_line=+3.0,
            closing_line=+4.5,
            bet_side="away"
        )
        
        assert clv < 0  # Got +3, closed at +4.5, bad for away bettor
    
    def test_clv_positive_total_over(self):
        """Betting over that moves up should be positive CLV."""
        from app.services.betting.clv import calculate_clv
        
        clv = calculate_clv(
            bet_line=210.0,
            closing_line=213.0,
            bet_side="over"
        )
        
        assert clv > 0
    
    def test_clv_zero_no_movement(self):
        """No line movement should result in zero CLV."""
        from app.services.betting.clv import calculate_clv
        
        clv = calculate_clv(
            bet_line=-5.0,
            closing_line=-5.0,
            bet_side="home"
        )
        
        assert clv == 0


class TestOddsConversion:
    """Test odds format conversions."""
    
    def test_american_to_decimal_positive(self):
        """Positive American odds conversion."""
        from app.services.betting.odds import american_to_decimal
        
        decimal_odds = american_to_decimal(150)
        assert abs(decimal_odds - 2.50) < 0.01
    
    def test_american_to_decimal_negative(self):
        """Negative American odds conversion."""
        from app.services.betting.odds import american_to_decimal
        
        decimal_odds = american_to_decimal(-150)
        assert abs(decimal_odds - 1.667) < 0.01
    
    def test_decimal_to_american_plus(self):
        """Decimal to American (plus odds)."""
        from app.services.betting.odds import decimal_to_american
        
        american_odds = decimal_to_american(2.50)
        assert american_odds == 150
    
    def test_decimal_to_american_minus(self):
        """Decimal to American (minus odds)."""
        from app.services.betting.odds import decimal_to_american
        
        american_odds = decimal_to_american(1.50)
        assert american_odds == -200
    
    def test_implied_probability_from_american_positive(self):
        """Implied probability from positive American odds."""
        from app.services.betting.odds import implied_probability
        
        prob = implied_probability(150)
        assert abs(prob - 0.40) < 0.01
    
    def test_implied_probability_from_american_negative(self):
        """Implied probability from negative American odds."""
        from app.services.betting.odds import implied_probability
        
        prob = implied_probability(-150)
        assert abs(prob - 0.60) < 0.01


class TestSignalTier:
    """Test signal tier classification."""
    
    def test_tier_a_classification(self):
        """Probability >= 0.65 should be Tier A."""
        from app.services.predictions.tiers import assign_signal_tier
        
        assert assign_signal_tier(0.65) == "A"
        assert assign_signal_tier(0.70) == "A"
        assert assign_signal_tier(0.80) == "A"
    
    def test_tier_b_classification(self):
        """Probability 0.60-0.65 should be Tier B."""
        from app.services.predictions.tiers import assign_signal_tier
        
        assert assign_signal_tier(0.60) == "B"
        assert assign_signal_tier(0.62) == "B"
        assert assign_signal_tier(0.649) == "B"
    
    def test_tier_c_classification(self):
        """Probability 0.55-0.60 should be Tier C."""
        from app.services.predictions.tiers import assign_signal_tier
        
        assert assign_signal_tier(0.55) == "C"
        assert assign_signal_tier(0.57) == "C"
        assert assign_signal_tier(0.599) == "C"
    
    def test_tier_d_classification(self):
        """Probability < 0.55 should be Tier D."""
        from app.services.predictions.tiers import assign_signal_tier
        
        assert assign_signal_tier(0.54) == "D"
        assert assign_signal_tier(0.50) == "D"
        assert assign_signal_tier(0.40) == "D"


class TestSHA256Verification:
    """Test SHA-256 prediction integrity."""
    
    def test_hash_generation(self):
        """Test hash is generated correctly."""
        from app.services.predictions.integrity import hash_prediction
        
        prediction_data = {
            "game_id": 12345,
            "bet_type": "spread",
            "predicted_side": "home",
            "probability": 0.65,
            "line": -3.5,
            "odds": -110,
            "timestamp": "2024-12-15T10:00:00Z"
        }
        
        hash_value = hash_prediction(prediction_data)
        
        assert len(hash_value) == 64
        assert all(c in "0123456789abcdef" for c in hash_value)
    
    def test_hash_consistency(self):
        """Same data should produce same hash."""
        from app.services.predictions.integrity import hash_prediction
        
        prediction_data = {
            "game_id": 12345,
            "bet_type": "spread",
            "predicted_side": "home",
            "probability": 0.65,
            "line": -3.5,
            "odds": -110,
            "timestamp": "2024-12-15T10:00:00Z"
        }
        
        hash1 = hash_prediction(prediction_data)
        hash2 = hash_prediction(prediction_data)
        
        assert hash1 == hash2
    
    def test_hash_changes_with_data(self):
        """Different data should produce different hash."""
        from app.services.predictions.integrity import hash_prediction
        
        prediction_data1 = {
            "game_id": 12345,
            "bet_type": "spread",
            "predicted_side": "home",
            "probability": 0.65,
            "line": -3.5,
            "odds": -110,
            "timestamp": "2024-12-15T10:00:00Z"
        }
        
        prediction_data2 = {
            "game_id": 12345,
            "bet_type": "spread",
            "predicted_side": "away",  # Changed
            "probability": 0.65,
            "line": -3.5,
            "odds": -110,
            "timestamp": "2024-12-15T10:00:00Z"
        }
        
        hash1 = hash_prediction(prediction_data1)
        hash2 = hash_prediction(prediction_data2)
        
        assert hash1 != hash2
    
    def test_hash_verification_valid(self):
        """Valid hash should verify correctly."""
        from app.services.predictions.integrity import (
            hash_prediction, verify_prediction_hash
        )
        
        prediction_data = {
            "game_id": 12345,
            "bet_type": "spread",
            "predicted_side": "home",
            "probability": 0.65,
            "line": -3.5,
            "odds": -110,
            "timestamp": "2024-12-15T10:00:00Z"
        }
        
        original_hash = hash_prediction(prediction_data)
        
        assert verify_prediction_hash(prediction_data, original_hash) is True
    
    def test_hash_verification_tampered(self):
        """Tampered data should fail verification."""
        from app.services.predictions.integrity import (
            hash_prediction, verify_prediction_hash
        )
        
        prediction_data = {
            "game_id": 12345,
            "bet_type": "spread",
            "predicted_side": "home",
            "probability": 0.65,
            "line": -3.5,
            "odds": -110,
            "timestamp": "2024-12-15T10:00:00Z"
        }
        
        original_hash = hash_prediction(prediction_data)
        
        # Tamper with data
        prediction_data["probability"] = 0.75
        
        assert verify_prediction_hash(prediction_data, original_hash) is False


class TestAutoGrading:
    """Test auto-grading logic."""
    
    def test_spread_win_home(self):
        """Home cover wins against spread."""
        from app.services.predictions.grading import grade_spread
        
        result = grade_spread(
            home_score=28,
            away_score=24,
            spread=-3.0,
            predicted_side="home"
        )
        
        assert result == "win"  # Won by 4, needed to cover 3
    
    def test_spread_loss_home(self):
        """Home fails to cover spread."""
        from app.services.predictions.grading import grade_spread
        
        result = grade_spread(
            home_score=26,
            away_score=24,
            spread=-3.0,
            predicted_side="home"
        )
        
        assert result == "loss"  # Won by 2, needed 3
    
    def test_spread_push(self):
        """Exact spread should be push."""
        from app.services.predictions.grading import grade_spread
        
        result = grade_spread(
            home_score=27,
            away_score=24,
            spread=-3.0,
            predicted_side="home"
        )
        
        assert result == "push"  # Won by exactly 3
    
    def test_moneyline_win(self):
        """Correct winner prediction."""
        from app.services.predictions.grading import grade_moneyline
        
        result = grade_moneyline(
            home_score=105,
            away_score=98,
            predicted_side="home"
        )
        
        assert result == "win"
    
    def test_moneyline_loss(self):
        """Wrong winner prediction."""
        from app.services.predictions.grading import grade_moneyline
        
        result = grade_moneyline(
            home_score=95,
            away_score=102,
            predicted_side="home"
        )
        
        assert result == "loss"
    
    def test_total_over_win(self):
        """Over prediction wins."""
        from app.services.predictions.grading import grade_total
        
        result = grade_total(
            home_score=110,
            away_score=108,
            total_line=215.0,
            predicted_side="over"
        )
        
        assert result == "win"  # Total 218 > 215
    
    def test_total_under_win(self):
        """Under prediction wins."""
        from app.services.predictions.grading import grade_total
        
        result = grade_total(
            home_score=100,
            away_score=105,
            total_line=210.0,
            predicted_side="under"
        )
        
        assert result == "win"  # Total 205 < 210
    
    def test_total_push(self):
        """Exact total is push."""
        from app.services.predictions.grading import grade_total
        
        result = grade_total(
            home_score=105,
            away_score=110,
            total_line=215.0,
            predicted_side="over"
        )
        
        assert result == "push"  # Total 215 = 215


class TestProfitCalculation:
    """Test profit/loss calculations."""
    
    def test_profit_positive_odds_win(self):
        """Win with positive odds."""
        from app.services.betting.profit import calculate_profit
        
        profit = calculate_profit(
            stake=100,
            american_odds=150,
            result="win"
        )
        
        assert profit == 150  # Win 1.5x stake
    
    def test_profit_negative_odds_win(self):
        """Win with negative odds."""
        from app.services.betting.profit import calculate_profit
        
        profit = calculate_profit(
            stake=150,
            american_odds=-150,
            result="win"
        )
        
        assert profit == 100  # Win 100 on 150 stake
    
    def test_profit_loss(self):
        """Loss returns negative stake."""
        from app.services.betting.profit import calculate_profit
        
        profit = calculate_profit(
            stake=100,
            american_odds=-110,
            result="loss"
        )
        
        assert profit == -100
    
    def test_profit_push(self):
        """Push returns zero."""
        from app.services.betting.profit import calculate_profit
        
        profit = calculate_profit(
            stake=100,
            american_odds=-110,
            result="push"
        )
        
        assert profit == 0


class TestPasswordHashing:
    """Test password security."""
    
    def test_password_hash_differs_from_plain(self):
        """Hash should not equal plaintext."""
        from app.core.security import get_password_hash
        
        password = "testpassword123"
        hashed = get_password_hash(password)
        
        assert hashed != password
    
    def test_password_verification_correct(self):
        """Correct password should verify."""
        from app.core.security import get_password_hash, verify_password
        
        password = "testpassword123"
        hashed = get_password_hash(password)
        
        assert verify_password(password, hashed) is True
    
    def test_password_verification_incorrect(self):
        """Incorrect password should not verify."""
        from app.core.security import get_password_hash, verify_password
        
        password = "testpassword123"
        hashed = get_password_hash(password)
        
        assert verify_password("wrongpassword", hashed) is False
    
    def test_password_hash_uniqueness(self):
        """Same password should produce different hashes (salt)."""
        from app.core.security import get_password_hash
        
        password = "testpassword123"
        hash1 = get_password_hash(password)
        hash2 = get_password_hash(password)
        
        assert hash1 != hash2  # Due to random salt


class TestAESEncryption:
    """Test AES encryption."""
    
    def test_encrypt_decrypt_roundtrip(self):
        """Data should survive encrypt/decrypt cycle."""
        from app.core.security import encrypt_data, decrypt_data
        
        original = "sensitive data"
        encrypted = encrypt_data(original)
        decrypted = decrypt_data(encrypted)
        
        assert decrypted == original
    
    def test_encrypted_differs_from_plain(self):
        """Encrypted data should not equal plaintext."""
        from app.core.security import encrypt_data
        
        original = "sensitive data"
        encrypted = encrypt_data(original)
        
        assert encrypted != original
    
    def test_different_data_different_ciphertext(self):
        """Different data should produce different ciphertext."""
        from app.core.security import encrypt_data
        
        encrypted1 = encrypt_data("data1")
        encrypted2 = encrypt_data("data2")
        
        assert encrypted1 != encrypted2


class TestJWTTokens:
    """Test JWT token handling."""
    
    def test_create_access_token(self):
        """Access token should be created."""
        from app.core.security import create_access_token
        
        token = create_access_token({"sub": "123", "role": "user"})
        
        assert token is not None
        assert len(token) > 50
    
    def test_verify_valid_token(self):
        """Valid token should verify."""
        from app.core.security import create_access_token, verify_token
        
        token = create_access_token({"sub": "123", "role": "user"})
        payload = verify_token(token)
        
        assert payload is not None
        assert payload["sub"] == "123"
        assert payload["role"] == "user"
    
    def test_verify_invalid_token(self):
        """Invalid token should not verify."""
        from app.core.security import verify_token
        
        payload = verify_token("invalid.token.here")
        
        assert payload is None
    
    def test_verify_expired_token(self):
        """Expired token should not verify."""
        from app.core.security import create_access_token, verify_token
        from datetime import timedelta
        
        token = create_access_token(
            {"sub": "123"},
            expires_delta=timedelta(seconds=-1)  # Already expired
        )
        
        payload = verify_token(token)
        
        assert payload is None


class TestCacheManager:
    """Test cache operations."""
    
    @pytest.mark.asyncio
    async def test_cache_set_get(self):
        """Basic cache set and get."""
        from app.core.cache import cache_manager
        
        with patch.object(cache_manager, '_redis') as mock_redis:
            mock_redis.set = AsyncMock(return_value=True)
            mock_redis.get = AsyncMock(return_value=b'"test_value"')
            
            await cache_manager.set("test_key", "test_value")
            value = await cache_manager.get("test_key")
            
            assert value == "test_value"
    
    @pytest.mark.asyncio
    async def test_cache_delete(self):
        """Cache delete operation."""
        from app.core.cache import cache_manager
        
        with patch.object(cache_manager, '_redis') as mock_redis:
            mock_redis.delete = AsyncMock(return_value=1)
            
            result = await cache_manager.delete("test_key")
            
            mock_redis.delete.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cache_ttl(self):
        """Cache with TTL."""
        from app.core.cache import cache_manager
        
        with patch.object(cache_manager, '_redis') as mock_redis:
            mock_redis.setex = AsyncMock(return_value=True)
            
            await cache_manager.set("test_key", "test_value", ttl=300)
            
            mock_redis.setex.assert_called_once()


class TestDataValidation:
    """Test data validation."""
    
    def test_odds_validation_valid(self):
        """Valid odds should pass."""
        from app.services.data_quality.validators import validate_odds
        
        odds_data = {
            "game_id": 12345,
            "sportsbook": "pinnacle",
            "spread": -3.5,
            "spread_odds": -110,
            "total": 220.5,
            "total_odds": -110
        }
        
        errors = validate_odds(odds_data)
        
        assert len(errors) == 0
    
    def test_odds_validation_invalid_spread(self):
        """Invalid spread should fail."""
        from app.services.data_quality.validators import validate_odds
        
        odds_data = {
            "game_id": 12345,
            "sportsbook": "pinnacle",
            "spread": -100.0,  # Invalid
            "spread_odds": -110
        }
        
        errors = validate_odds(odds_data)
        
        assert len(errors) > 0
    
    def test_game_validation_valid(self):
        """Valid game data should pass."""
        from app.services.data_quality.validators import validate_game
        
        game_data = {
            "external_id": "game_123",
            "sport_code": "NBA",
            "home_team_id": 1,
            "away_team_id": 2,
            "scheduled_at": "2024-12-15T19:00:00Z"
        }
        
        errors = validate_game(game_data)
        
        assert len(errors) == 0
    
    def test_game_validation_missing_required(self):
        """Missing required field should fail."""
        from app.services.data_quality.validators import validate_game
        
        game_data = {
            "sport_code": "NBA"
            # Missing required fields
        }
        
        errors = validate_game(game_data)
        
        assert len(errors) > 0


# Additional test fixtures
@pytest.fixture
def sample_prediction():
    """Sample prediction for testing."""
    return {
        "id": 1,
        "game_id": 12345,
        "sport_code": "NBA",
        "bet_type": "spread",
        "predicted_side": "home",
        "probability": 0.65,
        "edge": 0.08,
        "signal_tier": "A",
        "line": -3.5,
        "odds": -110,
        "created_at": datetime.utcnow()
    }


@pytest.fixture
def sample_game():
    """Sample game for testing."""
    return {
        "id": 12345,
        "external_id": "espn_12345",
        "sport_code": "NBA",
        "home_team_id": 1,
        "away_team_id": 2,
        "scheduled_at": datetime.utcnow() + timedelta(hours=3),
        "status": "scheduled"
    }


@pytest.fixture
def sample_odds():
    """Sample odds for testing."""
    return {
        "game_id": 12345,
        "sportsbook": "pinnacle",
        "spread_home": -3.5,
        "spread_away": 3.5,
        "spread_home_odds": -110,
        "spread_away_odds": -110,
        "total": 220.5,
        "over_odds": -110,
        "under_odds": -110,
        "ml_home": -150,
        "ml_away": 130,
        "recorded_at": datetime.utcnow()
    }
