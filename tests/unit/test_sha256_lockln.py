"""
Unit tests for SHA-256 Prediction Lock-In.
Tests cryptographic hashing, verification, and tamper detection.
"""

import pytest
import hashlib
import json
from datetime import datetime

from app.services.integrity.sha256_lockln import (
    hash_prediction,
    verify_prediction_hash,
    PredictionHasher,
    PredictionIntegrity,
    generate_prediction_receipt,
)


class TestHashPrediction:
    """Tests for hash_prediction function."""
    
    @pytest.fixture
    def sample_prediction(self):
        """Create a sample prediction for testing."""
        return {
            "game_id": "game_12345",
            "bet_type": "spread",
            "predicted_side": "home",
            "probability": 0.6523456,
            "line_at_prediction": -3.5,
            "odds_at_prediction": -110,
            "locked_at": "2024-03-15T14:30:00Z",
        }
    
    def test_hash_is_sha256(self, sample_prediction):
        """Test that hash is SHA-256 format (64 hex characters)."""
        hash_value = hash_prediction(sample_prediction)
        assert len(hash_value) == 64
        assert all(c in '0123456789abcdef' for c in hash_value)
    
    def test_hash_consistency(self, sample_prediction):
        """Test that same prediction produces same hash."""
        hash1 = hash_prediction(sample_prediction)
        hash2 = hash_prediction(sample_prediction)
        assert hash1 == hash2
    
    def test_hash_changes_with_data(self, sample_prediction):
        """Test that different data produces different hash."""
        hash1 = hash_prediction(sample_prediction)
        
        modified = sample_prediction.copy()
        modified["probability"] = 0.70
        hash2 = hash_prediction(modified)
        
        assert hash1 != hash2
    
    def test_hash_with_hmac(self, sample_prediction):
        """Test HMAC-based hashing with secret key."""
        secret_key = "my_secret_key_123"
        hash1 = hash_prediction(sample_prediction, secret_key=secret_key)
        hash2 = hash_prediction(sample_prediction, secret_key=secret_key)
        
        assert hash1 == hash2
        
        # Different key produces different hash
        hash3 = hash_prediction(sample_prediction, secret_key="different_key")
        assert hash1 != hash3
    
    def test_hash_without_optional_fields(self):
        """Test hashing with minimal required fields."""
        minimal_prediction = {
            "game_id": "game_001",
            "bet_type": "moneyline",
            "predicted_side": "away",
            "probability": 0.55,
            "line_at_prediction": 0,
            "odds_at_prediction": 150,
            "locked_at": "2024-01-01T00:00:00Z",
        }
        
        hash_value = hash_prediction(minimal_prediction)
        assert len(hash_value) == 64
    
    def test_hash_with_optional_fields(self):
        """Test hashing with all optional fields."""
        full_prediction = {
            "game_id": "game_002",
            "bet_type": "total",
            "predicted_side": "over",
            "probability": 0.62,
            "line_at_prediction": 220.5,
            "odds_at_prediction": -105,
            "locked_at": "2024-02-15T10:00:00Z",
            "sport_code": "NBA",
            "signal_tier": "A",
            "model_id": "model_v2.1",
        }
        
        hash_value = hash_prediction(full_prediction)
        assert len(hash_value) == 64


class TestVerifyPredictionHash:
    """Tests for verify_prediction_hash function."""
    
    @pytest.fixture
    def prediction_with_hash(self):
        """Create prediction with its hash."""
        prediction = {
            "game_id": "game_verify_001",
            "bet_type": "spread",
            "predicted_side": "home",
            "probability": 0.65,
            "line_at_prediction": -5.5,
            "odds_at_prediction": -110,
            "locked_at": "2024-03-20T16:00:00Z",
        }
        stored_hash = hash_prediction(prediction)
        return prediction, stored_hash
    
    def test_valid_prediction_passes(self, prediction_with_hash):
        """Test that unmodified prediction passes verification."""
        prediction, stored_hash = prediction_with_hash
        is_valid = verify_prediction_hash(prediction, stored_hash)
        assert is_valid is True
    
    def test_modified_prediction_fails(self, prediction_with_hash):
        """Test that modified prediction fails verification."""
        prediction, stored_hash = prediction_with_hash
        
        # Modify the prediction
        prediction["probability"] = 0.70
        
        is_valid = verify_prediction_hash(prediction, stored_hash)
        assert is_valid is False
    
    def test_wrong_hash_fails(self, prediction_with_hash):
        """Test that wrong hash fails verification."""
        prediction, _ = prediction_with_hash
        wrong_hash = "a" * 64  # Invalid hash
        
        is_valid = verify_prediction_hash(prediction, wrong_hash)
        assert is_valid is False
    
    def test_verification_with_hmac(self, prediction_with_hash):
        """Test verification with HMAC."""
        prediction, _ = prediction_with_hash
        secret_key = "verification_key"
        
        # Generate HMAC hash
        hmac_hash = hash_prediction(prediction, secret_key=secret_key)
        
        # Verify with same key
        is_valid = verify_prediction_hash(
            prediction, hmac_hash, secret_key=secret_key
        )
        assert is_valid is True
        
        # Verify with wrong key fails
        is_valid = verify_prediction_hash(
            prediction, hmac_hash, secret_key="wrong_key"
        )
        assert is_valid is False


class TestPredictionIntegrity:
    """Tests for PredictionIntegrity dataclass."""
    
    def test_valid_integrity(self):
        """Test valid integrity result."""
        integrity = PredictionIntegrity(
            prediction_id="pred_001",
            original_hash="a" * 64,
            computed_hash="a" * 64,
            is_valid=True,
            verified_at=datetime.now(),
            mismatch_fields=[],
        )
        
        assert integrity.is_valid is True
        assert integrity.mismatch_fields == []
    
    def test_invalid_integrity(self):
        """Test invalid integrity result with mismatch."""
        integrity = PredictionIntegrity(
            prediction_id="pred_002",
            original_hash="a" * 64,
            computed_hash="b" * 64,
            is_valid=False,
            verified_at=datetime.now(),
            mismatch_fields=["probability"],
        )
        
        assert integrity.is_valid is False
        assert "probability" in integrity.mismatch_fields


class TestPredictionHasher:
    """Tests for PredictionHasher class."""
    
    @pytest.fixture
    def hasher(self):
        """Create a PredictionHasher instance."""
        return PredictionHasher()
    
    @pytest.fixture
    def hasher_with_key(self):
        """Create a PredictionHasher with secret key."""
        return PredictionHasher(secret_key="test_secret_key")
    
    def test_hash_single(self, hasher):
        """Test hashing a single prediction."""
        prediction = {
            "game_id": "game_100",
            "bet_type": "spread",
            "predicted_side": "home",
            "probability": 0.62,
            "line_at_prediction": -7.0,
            "odds_at_prediction": -110,
            "locked_at": "2024-04-01T12:00:00Z",
        }
        
        hash_value = hasher.hash(prediction)
        assert len(hash_value) == 64
    
    def test_verify_single(self, hasher):
        """Test verifying a single prediction."""
        prediction = {
            "game_id": "game_101",
            "bet_type": "moneyline",
            "predicted_side": "away",
            "probability": 0.58,
            "line_at_prediction": 0,
            "odds_at_prediction": 120,
            "locked_at": "2024-04-02T14:00:00Z",
        }
        
        hash_value = hasher.hash(prediction)
        is_valid = hasher.verify(prediction, hash_value)
        assert is_valid is True
    
    def test_batch_hash(self, hasher):
        """Test batch hashing multiple predictions."""
        predictions = [
            {
                "game_id": f"game_{i}",
                "bet_type": "spread",
                "predicted_side": "home",
                "probability": 0.60 + i * 0.01,
                "line_at_prediction": -3.0,
                "odds_at_prediction": -110,
                "locked_at": "2024-04-03T10:00:00Z",
            }
            for i in range(5)
        ]
        
        hashes = hasher.batch_hash(predictions)
        assert len(hashes) == 5
        assert all(len(h) == 64 for h in hashes)
        
        # All hashes should be unique
        assert len(set(hashes)) == 5
    
    def test_batch_verify(self, hasher):
        """Test batch verification."""
        predictions = [
            {
                "game_id": f"game_{i}",
                "bet_type": "total",
                "predicted_side": "over",
                "probability": 0.55,
                "line_at_prediction": 200.0 + i,
                "odds_at_prediction": -105,
                "locked_at": "2024-04-04T09:00:00Z",
            }
            for i in range(3)
        ]
        
        hashes = hasher.batch_hash(predictions)
        results = hasher.batch_verify(predictions, hashes)
        
        assert all(results)
    
    def test_hasher_with_secret_key(self, hasher_with_key):
        """Test hasher with secret key produces HMAC."""
        prediction = {
            "game_id": "game_secret",
            "bet_type": "spread",
            "predicted_side": "away",
            "probability": 0.64,
            "line_at_prediction": 2.5,
            "odds_at_prediction": -108,
            "locked_at": "2024-04-05T11:00:00Z",
        }
        
        hash_value = hasher_with_key.hash(prediction)
        is_valid = hasher_with_key.verify(prediction, hash_value)
        
        assert is_valid is True


class TestGeneratePredictionReceipt:
    """Tests for generate_prediction_receipt function."""
    
    def test_receipt_generation(self):
        """Test receipt generation."""
        prediction = {
            "game_id": "game_receipt_001",
            "bet_type": "spread",
            "predicted_side": "home",
            "probability": 0.67,
            "line_at_prediction": -4.5,
            "odds_at_prediction": -110,
            "locked_at": "2024-04-10T15:30:00Z",
        }
        hash_value = hash_prediction(prediction)
        
        receipt = generate_prediction_receipt(prediction, hash_value)
        
        assert "game_receipt_001" in receipt
        assert "spread" in receipt
        assert hash_value in receipt
    
    def test_receipt_format(self):
        """Test receipt has proper format."""
        prediction = {
            "game_id": "game_format_001",
            "bet_type": "moneyline",
            "predicted_side": "away",
            "probability": 0.55,
            "line_at_prediction": 0,
            "odds_at_prediction": 140,
            "locked_at": "2024-04-11T08:00:00Z",
        }
        hash_value = hash_prediction(prediction)
        
        receipt = generate_prediction_receipt(prediction, hash_value)
        
        # Should contain key information
        assert "Game ID" in receipt or "game_id" in receipt
        assert "Probability" in receipt or "probability" in receipt
        assert "Hash" in receipt or "SHA-256" in receipt


class TestCanonicalData:
    """Tests for canonical data creation."""
    
    def test_probability_rounding(self):
        """Test probability is rounded to 6 decimals."""
        pred1 = {
            "game_id": "game_round",
            "bet_type": "spread",
            "predicted_side": "home",
            "probability": 0.65234567890,  # More than 6 decimals
            "line_at_prediction": -3.5,
            "odds_at_prediction": -110,
            "locked_at": "2024-04-15T12:00:00Z",
        }
        
        pred2 = {
            "game_id": "game_round",
            "bet_type": "spread",
            "predicted_side": "home",
            "probability": 0.652346,  # Rounded to 6 decimals
            "line_at_prediction": -3.5,
            "odds_at_prediction": -110,
            "locked_at": "2024-04-15T12:00:00Z",
        }
        
        hash1 = hash_prediction(pred1)
        hash2 = hash_prediction(pred2)
        
        # Should produce same hash after rounding
        assert hash1 == hash2
    
    def test_line_rounding(self):
        """Test line is rounded to 2 decimals."""
        pred1 = {
            "game_id": "game_line",
            "bet_type": "total",
            "predicted_side": "over",
            "probability": 0.58,
            "line_at_prediction": 220.5555,  # More than 2 decimals
            "odds_at_prediction": -110,
            "locked_at": "2024-04-16T10:00:00Z",
        }
        
        pred2 = {
            "game_id": "game_line",
            "bet_type": "total",
            "predicted_side": "over",
            "probability": 0.58,
            "line_at_prediction": 220.56,  # Rounded to 2 decimals
            "odds_at_prediction": -110,
            "locked_at": "2024-04-16T10:00:00Z",
        }
        
        hash1 = hash_prediction(pred1)
        hash2 = hash_prediction(pred2)
        
        # Should produce same hash after rounding
        assert hash1 == hash2
    
    def test_key_ordering(self):
        """Test hash is consistent regardless of key order."""
        pred1 = {
            "game_id": "game_order",
            "bet_type": "spread",
            "predicted_side": "home",
            "probability": 0.60,
            "line_at_prediction": -5.0,
            "odds_at_prediction": -110,
            "locked_at": "2024-04-17T14:00:00Z",
        }
        
        # Same data, different key order
        pred2 = {
            "locked_at": "2024-04-17T14:00:00Z",
            "odds_at_prediction": -110,
            "line_at_prediction": -5.0,
            "probability": 0.60,
            "predicted_side": "home",
            "bet_type": "spread",
            "game_id": "game_order",
        }
        
        hash1 = hash_prediction(pred1)
        hash2 = hash_prediction(pred2)
        
        # Should produce same hash regardless of key order
        assert hash1 == hash2


class TestTamperDetection:
    """Tests for tamper detection scenarios."""
    
    def test_detect_probability_change(self):
        """Test detection of probability tampering."""
        original = {
            "game_id": "game_tamper_prob",
            "bet_type": "spread",
            "predicted_side": "home",
            "probability": 0.55,  # Original: 55%
            "line_at_prediction": -3.0,
            "odds_at_prediction": -110,
            "locked_at": "2024-04-20T12:00:00Z",
        }
        
        original_hash = hash_prediction(original)
        
        # Tamper: Change probability to 65%
        tampered = original.copy()
        tampered["probability"] = 0.65
        
        is_valid = verify_prediction_hash(tampered, original_hash)
        assert is_valid is False
    
    def test_detect_side_change(self):
        """Test detection of predicted side tampering."""
        original = {
            "game_id": "game_tamper_side",
            "bet_type": "spread",
            "predicted_side": "away",  # Original: away
            "probability": 0.60,
            "line_at_prediction": 3.0,
            "odds_at_prediction": -110,
            "locked_at": "2024-04-21T14:00:00Z",
        }
        
        original_hash = hash_prediction(original)
        
        # Tamper: Change to home
        tampered = original.copy()
        tampered["predicted_side"] = "home"
        
        is_valid = verify_prediction_hash(tampered, original_hash)
        assert is_valid is False
    
    def test_detect_timestamp_change(self):
        """Test detection of timestamp tampering."""
        original = {
            "game_id": "game_tamper_time",
            "bet_type": "moneyline",
            "predicted_side": "home",
            "probability": 0.58,
            "line_at_prediction": 0,
            "odds_at_prediction": -150,
            "locked_at": "2024-04-22T09:00:00Z",  # Original time
        }
        
        original_hash = hash_prediction(original)
        
        # Tamper: Change timestamp (trying to claim earlier prediction)
        tampered = original.copy()
        tampered["locked_at"] = "2024-04-21T09:00:00Z"
        
        is_valid = verify_prediction_hash(tampered, original_hash)
        assert is_valid is False
