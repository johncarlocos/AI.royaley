"""
LOYALEY - Security Unit Tests
Phase 1: Test Security Components
"""

import pytest
from datetime import datetime, timedelta

from app.core.security import (
    PasswordHasher,
    TOTPManager,
    Encryptor,
    generate_api_key,
    hash_api_key,
    hash_prediction,
)


class TestPasswordHasher:
    """Test password hashing functionality."""
    
    def test_hash_password(self, password_hasher: PasswordHasher):
        """Test password hashing."""
        password = "testpassword123"
        hashed = password_hasher.hash(password)
        
        assert hashed != password
        assert len(hashed) > 0
        assert password_hasher.verify(password, hashed)
    
    def test_verify_wrong_password(self, password_hasher: PasswordHasher):
        """Test wrong password verification."""
        password = "testpassword123"
        hashed = password_hasher.hash(password)
        
        assert not password_hasher.verify("wrongpassword", hashed)
    
    def test_hash_uniqueness(self, password_hasher: PasswordHasher):
        """Test that same password produces different hashes."""
        password = "testpassword123"
        hash1 = password_hasher.hash(password)
        hash2 = password_hasher.hash(password)
        
        # Hashes should be different (due to salt)
        assert hash1 != hash2
        # But both should verify correctly
        assert password_hasher.verify(password, hash1)
        assert password_hasher.verify(password, hash2)


class TestTOTPManager:
    """Test TOTP two-factor authentication."""
    
    def test_generate_secret(self):
        """Test secret generation."""
        totp = TOTPManager()
        secret = totp.generate_secret()
        
        assert secret is not None
        assert len(secret) == 32  # Base32 encoded
    
    def test_verify_token(self):
        """Test token verification."""
        totp = TOTPManager()
        secret = totp.generate_secret()
        
        # Generate current token
        import pyotp
        current_token = pyotp.TOTP(secret).now()
        
        assert totp.verify_token(secret, current_token)
    
    def test_verify_wrong_token(self):
        """Test wrong token rejection."""
        totp = TOTPManager()
        secret = totp.generate_secret()
        
        assert not totp.verify_token(secret, "000000")
    
    def test_get_provisioning_uri(self):
        """Test QR code URI generation."""
        totp = TOTPManager()
        secret = totp.generate_secret()
        
        uri = totp.get_provisioning_uri(secret, "test@example.com")
        
        assert "otpauth://totp/" in uri
        assert "AI%20PRO%20SPORTS" in uri
        assert "test%40example.com" in uri


class TestEncryptor:
    """Test AES encryption."""
    
    def test_encrypt_decrypt(self):
        """Test encryption and decryption."""
        encryptor = Encryptor()
        plaintext = "sensitive data here"
        
        encrypted = encryptor.encrypt(plaintext)
        decrypted = encryptor.decrypt(encrypted)
        
        assert encrypted != plaintext
        assert decrypted == plaintext
    
    def test_encrypt_uniqueness(self):
        """Test that same plaintext produces different ciphertext."""
        encryptor = Encryptor()
        plaintext = "sensitive data"
        
        encrypted1 = encryptor.encrypt(plaintext)
        encrypted2 = encryptor.encrypt(plaintext)
        
        # Encryptions should be different (due to IV)
        assert encrypted1 != encrypted2
        # But both should decrypt correctly
        assert encryptor.decrypt(encrypted1) == plaintext
        assert encryptor.decrypt(encrypted2) == plaintext


class TestAPIKey:
    """Test API key generation."""
    
    def test_generate_api_key(self):
        """Test API key generation."""
        key, prefix = generate_api_key()
        
        assert key is not None
        assert len(key) > 20
        assert prefix is not None
        assert len(prefix) == 8
        assert key.startswith(prefix)
    
    def test_hash_api_key(self):
        """Test API key hashing."""
        key, _ = generate_api_key()
        hashed = hash_api_key(key)
        
        assert hashed != key
        assert len(hashed) == 64  # SHA-256 hex digest


class TestPredictionHash:
    """Test prediction integrity hashing."""
    
    def test_hash_prediction(self):
        """Test prediction hash generation."""
        prediction_data = {
            "game_id": "game123",
            "bet_type": "spread",
            "predicted_side": "home",
            "probability": 0.65,
            "line": -3.5,
            "odds": -110,
            "timestamp": "2025-12-01T12:00:00",
        }
        
        hash1 = hash_prediction(prediction_data)
        hash2 = hash_prediction(prediction_data)
        
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex digest
    
    def test_hash_changes_on_modification(self):
        """Test that hash changes when data is modified."""
        prediction_data = {
            "game_id": "game123",
            "bet_type": "spread",
            "predicted_side": "home",
            "probability": 0.65,
            "line": -3.5,
            "odds": -110,
            "timestamp": "2025-12-01T12:00:00",
        }
        
        hash1 = hash_prediction(prediction_data)
        
        # Modify probability
        prediction_data["probability"] = 0.66
        hash2 = hash_prediction(prediction_data)
        
        assert hash1 != hash2
