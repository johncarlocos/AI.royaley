"""
ROYALEY - Phase 4 Enterprise Security
Complete security implementation: JWT, 2FA, AES-256 encryption, rate limiting
"""

import base64
import hashlib
import hmac
import json
import secrets
import time
import struct
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple
from enum import Enum

import jwt
import bcrypt
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding
from pydantic import BaseModel

from app.core.config import settings


class TokenType(str, Enum):
    """Token types for JWT"""
    ACCESS = "access"
    REFRESH = "refresh"
    API_KEY = "api_key"


class TokenData(BaseModel):
    """Token payload data"""
    user_id: int
    username: str
    email: str
    role: str
    token_type: TokenType
    exp: datetime
    iat: datetime
    jti: str  # JWT ID for revocation


class SecurityManager:
    """Enterprise security manager with comprehensive authentication"""
    
    def __init__(self):
        self.secret_key = settings.SECRET_KEY
        self.jwt_secret = settings.JWT_SECRET_KEY
        self.algorithm = settings.JWT_ALGORITHM
        self.access_token_expire = settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES
        self.refresh_token_expire = settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS
        self.bcrypt_rounds = settings.BCRYPT_ROUNDS
        self._revoked_tokens: set = set()
    
    # ===== Password Hashing =====
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt with configured rounds"""
        salt = bcrypt.gensalt(rounds=self.bcrypt_rounds)
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash"""
        try:
            return bcrypt.checkpw(
                password.encode('utf-8'),
                hashed.encode('utf-8')
            )
        except Exception:
            return False
    
    def check_password_strength(self, password: str) -> Tuple[bool, list]:
        """
        Check password meets security requirements
        Returns (is_valid, list of issues)
        """
        issues = []
        
        if len(password) < 8:
            issues.append("Password must be at least 8 characters")
        if len(password) > 128:
            issues.append("Password must not exceed 128 characters")
        if not any(c.isupper() for c in password):
            issues.append("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in password):
            issues.append("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in password):
            issues.append("Password must contain at least one digit")
        if not any(c in '!@#$%^&*()_+-=[]{}|;:,.<>?' for c in password):
            issues.append("Password must contain at least one special character")
        
        return len(issues) == 0, issues
    
    # ===== JWT Token Management =====
    
    def create_access_token(
        self,
        user_id: str,
        username: str,
        email: str,
        role: str = "user"
    ) -> str:
        """Create JWT access token"""
        now = datetime.now(timezone.utc)
        expire = now + timedelta(minutes=self.access_token_expire)
        
        payload = {
            "sub": user_id,  # subject for compatibility with dependencies
            "user_id": user_id,
            "username": username,
            "email": email,
            "role": role,
            "token_type": TokenType.ACCESS.value,
            "exp": expire,
            "iat": now,
            "jti": secrets.token_urlsafe(16)
        }
        
        return jwt.encode(payload, self.jwt_secret, algorithm=self.algorithm)
    
    def create_refresh_token(
        self,
        user_id: str,
        username: str,
        email: str,
        role: str = "user"
    ) -> str:
        """Create JWT refresh token"""
        now = datetime.now(timezone.utc)
        expire = now + timedelta(days=self.refresh_token_expire)
        
        payload = {
            "sub": user_id,  # subject for compatibility with dependencies
            "user_id": user_id,
            "username": username,
            "email": email,
            "role": role,
            "token_type": TokenType.REFRESH.value,
            "exp": expire,
            "iat": now,
            "jti": secrets.token_urlsafe(16)
        }
        
        return jwt.encode(payload, self.jwt_secret, algorithm=self.algorithm)
    
    def decode_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Decode and validate JWT token"""
        try:
            payload = jwt.decode(
                token,
                self.jwt_secret,
                algorithms=[self.algorithm]
            )
            
            # Check if token is revoked
            if payload.get("jti") in self._revoked_tokens:
                return None
            
            return payload
        except jwt.ExpiredSignatureError as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Token expired: {e}")
            return None
        except jwt.InvalidTokenError as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Invalid token: {e}")
            return None
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Token decode error: {e}")
            return None
    
    def revoke_token(self, token: str) -> bool:
        """Revoke a JWT token by its JTI"""
        try:
            payload = jwt.decode(
                token,
                self.jwt_secret,
                algorithms=[self.algorithm],
                options={"verify_exp": False}
            )
            jti = payload.get("jti")
            if jti:
                self._revoked_tokens.add(jti)
                return True
        except jwt.InvalidTokenError:
            pass
        return False
    
    def refresh_access_token(self, refresh_token: str) -> Optional[str]:
        """Generate new access token from refresh token"""
        payload = self.decode_token(refresh_token)
        
        if not payload:
            return None
        
        if payload.get("token_type") != TokenType.REFRESH.value:
            return None
        
        return self.create_access_token(
            user_id=payload["user_id"],
            username=payload["username"],
            email=payload["email"],
            role=payload["role"]
        )
    
    # ===== API Key Management =====
    
    def generate_api_key(self) -> str:
        """Generate a secure API key"""
        return f"aips_{secrets.token_urlsafe(32)}"
    
    def hash_api_key(self, api_key: str) -> str:
        """Hash API key for storage"""
        return hashlib.sha256(api_key.encode()).hexdigest()
    
    def verify_api_key(self, api_key: str, hashed_key: str) -> bool:
        """Verify API key against stored hash"""
        return hmac.compare_digest(
            self.hash_api_key(api_key),
            hashed_key
        )


class TOTPManager:
    """TOTP-based Two-Factor Authentication manager"""
    
    def __init__(self):
        self.issuer = settings.TOTP_ISSUER
        self.time_step = settings.TOTP_TIME_STEP
        self.digits = 6
        self.algorithm = "SHA1"
    
    def generate_secret(self) -> str:
        """Generate a new TOTP secret"""
        return base64.b32encode(secrets.token_bytes(20)).decode('utf-8')
    
    def get_provisioning_uri(self, secret: str, username: str) -> str:
        """Generate provisioning URI for QR code"""
        params = {
            "secret": secret,
            "issuer": self.issuer,
            "algorithm": self.algorithm,
            "digits": self.digits,
            "period": self.time_step
        }
        query = "&".join(f"{k}={v}" for k, v in params.items())
        return f"otpauth://totp/{self.issuer}:{username}?{query}"
    
    def generate_totp(self, secret: str, timestamp: Optional[float] = None) -> str:
        """Generate TOTP code for given timestamp"""
        if timestamp is None:
            timestamp = time.time()
        
        counter = int(timestamp // self.time_step)
        secret_bytes = base64.b32decode(secret.upper())
        
        # HMAC-SHA1
        counter_bytes = struct.pack(">Q", counter)
        hmac_hash = hmac.new(secret_bytes, counter_bytes, hashlib.sha1).digest()
        
        # Dynamic truncation
        offset = hmac_hash[-1] & 0x0F
        code = struct.unpack(">I", hmac_hash[offset:offset + 4])[0]
        code = (code & 0x7FFFFFFF) % (10 ** self.digits)
        
        return str(code).zfill(self.digits)
    
    def verify_totp(
        self,
        secret: str,
        code: str,
        window: int = 1
    ) -> bool:
        """
        Verify TOTP code with time window tolerance
        window=1 allows codes from previous and next time step
        """
        if len(code) != self.digits:
            return False
        
        timestamp = time.time()
        
        for i in range(-window, window + 1):
            test_time = timestamp + (i * self.time_step)
            expected_code = self.generate_totp(secret, test_time)
            if hmac.compare_digest(code, expected_code):
                return True
        
        return False


class AESEncryption:
    """AES-256 encryption for sensitive data at rest"""
    
    def __init__(self, key: Optional[str] = None):
        key_str = key or settings.AES_KEY
        # Ensure key is 32 bytes (256 bits)
        self.key = hashlib.sha256(key_str.encode()).digest()
        self.backend = default_backend()
    
    def encrypt(self, plaintext: str) -> str:
        """
        Encrypt plaintext using AES-256-CBC
        Returns base64-encoded ciphertext with IV prepended
        """
        iv = secrets.token_bytes(16)
        cipher = Cipher(
            algorithms.AES(self.key),
            modes.CBC(iv),
            backend=self.backend
        )
        
        # Pad plaintext to block size
        padder = padding.PKCS7(128).padder()
        padded_data = padder.update(plaintext.encode()) + padder.finalize()
        
        # Encrypt
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(padded_data) + encryptor.finalize()
        
        # Prepend IV and encode
        return base64.b64encode(iv + ciphertext).decode('utf-8')
    
    def decrypt(self, encrypted: str) -> str:
        """
        Decrypt base64-encoded ciphertext
        Expects IV prepended to ciphertext
        """
        data = base64.b64decode(encrypted.encode('utf-8'))
        iv = data[:16]
        ciphertext = data[16:]
        
        cipher = Cipher(
            algorithms.AES(self.key),
            modes.CBC(iv),
            backend=self.backend
        )
        
        # Decrypt
        decryptor = cipher.decryptor()
        padded_data = decryptor.update(ciphertext) + decryptor.finalize()
        
        # Remove padding
        unpadder = padding.PKCS7(128).unpadder()
        plaintext = unpadder.update(padded_data) + unpadder.finalize()
        
        return plaintext.decode('utf-8')
    
    def encrypt_dict(self, data: Dict[str, Any]) -> str:
        """Encrypt dictionary as JSON"""
        return self.encrypt(json.dumps(data))
    
    def decrypt_dict(self, encrypted: str) -> Dict[str, Any]:
        """Decrypt to dictionary"""
        return json.loads(self.decrypt(encrypted))


class SHA256Hasher:
    """SHA-256 hashing for prediction integrity verification"""
    
    @staticmethod
    def hash_prediction(prediction_data: Dict[str, Any]) -> str:
        """
        Generate SHA-256 hash for prediction lock-in
        Ensures prediction integrity cannot be tampered with
        """
        # Create canonical representation
        canonical_data = {
            "game_id": prediction_data.get("game_id"),
            "bet_type": prediction_data.get("bet_type"),
            "predicted_side": prediction_data.get("predicted_side"),
            "probability": round(prediction_data.get("probability", 0), 6),
            "line_at_prediction": prediction_data.get("line"),
            "odds_at_prediction": prediction_data.get("odds"),
            "locked_at": prediction_data.get("timestamp")
        }
        
        # Sort keys for consistency
        json_str = json.dumps(canonical_data, sort_keys=True)
        
        return hashlib.sha256(json_str.encode()).hexdigest()
    
    @staticmethod
    def verify_prediction(prediction_data: Dict[str, Any], stored_hash: str) -> bool:
        """Verify prediction integrity using constant-time comparison"""
        computed_hash = SHA256Hasher.hash_prediction(prediction_data)
        return hmac.compare_digest(computed_hash, stored_hash)
    
    @staticmethod
    def hash_string(data: str) -> str:
        """Generic string hashing"""
        return hashlib.sha256(data.encode()).hexdigest()


class RateLimiter:
    """Token bucket rate limiter"""
    
    def __init__(
        self,
        max_requests: int = 100,
        window_seconds: int = 60
    ):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._buckets: Dict[str, Dict] = {}
    
    def is_allowed(self, identifier: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if request is allowed for identifier
        Returns (is_allowed, rate_limit_info)
        """
        now = time.time()
        
        if identifier not in self._buckets:
            self._buckets[identifier] = {
                "tokens": self.max_requests - 1,
                "last_update": now
            }
            return True, self._get_limit_info(identifier)
        
        bucket = self._buckets[identifier]
        elapsed = now - bucket["last_update"]
        
        # Replenish tokens
        tokens_to_add = (elapsed / self.window_seconds) * self.max_requests
        bucket["tokens"] = min(self.max_requests, bucket["tokens"] + tokens_to_add)
        bucket["last_update"] = now
        
        if bucket["tokens"] >= 1:
            bucket["tokens"] -= 1
            return True, self._get_limit_info(identifier)
        
        return False, self._get_limit_info(identifier)
    
    def _get_limit_info(self, identifier: str) -> Dict[str, Any]:
        """Get rate limit info for response headers"""
        bucket = self._buckets.get(identifier, {"tokens": self.max_requests})
        return {
            "X-RateLimit-Limit": self.max_requests,
            "X-RateLimit-Remaining": int(bucket["tokens"]),
            "X-RateLimit-Reset": int(time.time()) + self.window_seconds
        }
    
    def reset(self, identifier: str):
        """Reset rate limit for identifier"""
        if identifier in self._buckets:
            del self._buckets[identifier]


# Global instances
security_manager = SecurityManager()
totp_manager = TOTPManager()
aes_encryption = AESEncryption()
sha256_hasher = SHA256Hasher()
rate_limiter = RateLimiter(
    max_requests=settings.RATE_LIMIT_REQUESTS,
    window_seconds=settings.RATE_LIMIT_WINDOW
)


def get_security_manager() -> SecurityManager:
    """Get the global security manager instance."""
    return security_manager


# FastAPI dependency functions - these are imported by routes
# The actual implementation uses app.api.dependencies but these 
# provide the interface that routes expect

async def get_current_user():
    """
    Placeholder for FastAPI dependency injection.
    Actual implementation is in app.api.dependencies.
    This function should not be called directly - use the dependency from dependencies.py
    """
    raise NotImplementedError(
        "Use 'from app.api.dependencies import get_current_user' instead"
    )


def require_roles(*roles):
    """
    Placeholder for FastAPI dependency injection.
    Actual implementation is in app.api.dependencies.
    """
    raise NotImplementedError(
        "Use 'from app.api.dependencies import require_roles' instead"
    )

