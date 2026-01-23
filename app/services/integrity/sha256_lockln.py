"""
SHA-256 Prediction Lock-In for LOYALEY.

Cryptographically hashes predictions at creation time to verify
integrity and prevent tampering.
"""

import hashlib
import hmac
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any
from uuid import UUID

logger = logging.getLogger(__name__)


@dataclass
class PredictionIntegrity:
    """Result of prediction integrity verification."""
    prediction_id: UUID
    original_hash: str
    computed_hash: str
    is_valid: bool
    verified_at: datetime
    mismatch_fields: list
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "prediction_id": str(self.prediction_id),
            "original_hash": self.original_hash,
            "computed_hash": self.computed_hash,
            "is_valid": self.is_valid,
            "verified_at": self.verified_at.isoformat(),
            "mismatch_fields": self.mismatch_fields,
        }


def hash_prediction(
    prediction_data: Dict[str, Any],
    timestamp: Optional[datetime] = None,
    secret_key: Optional[str] = None,
) -> str:
    """
    Generate SHA-256 hash for a prediction.
    
    Creates a cryptographic hash of prediction data to lock in
    the prediction at creation time and verify integrity later.
    
    Args:
        prediction_data: Dictionary containing prediction fields
        timestamp: Lock-in timestamp (defaults to now)
        secret_key: Optional HMAC secret for additional security
        
    Returns:
        SHA-256 hash as hexadecimal string
    """
    # Create canonical data structure with sorted keys
    canonical_data = _create_canonical_data(prediction_data, timestamp)
    
    # Convert to JSON string with sorted keys
    json_str = json.dumps(canonical_data, sort_keys=True, separators=(',', ':'))
    
    if secret_key:
        # Use HMAC for keyed hash
        hash_obj = hmac.new(
            secret_key.encode('utf-8'),
            json_str.encode('utf-8'),
            hashlib.sha256
        )
        return hash_obj.hexdigest()
    else:
        # Use plain SHA-256
        return hashlib.sha256(json_str.encode('utf-8')).hexdigest()


def verify_prediction_hash(
    prediction_data: Dict[str, Any],
    stored_hash: str,
    timestamp: datetime,
    secret_key: Optional[str] = None,
) -> PredictionIntegrity:
    """
    Verify a prediction's integrity by comparing hashes.
    
    Recomputes the hash from stored data and compares with
    the original hash using constant-time comparison to
    prevent timing attacks.
    
    Args:
        prediction_data: Dictionary containing prediction fields
        stored_hash: Original hash to compare against
        timestamp: Original lock-in timestamp
        secret_key: Optional HMAC secret
        
    Returns:
        PredictionIntegrity result
    """
    # Recompute hash
    computed_hash = hash_prediction(prediction_data, timestamp, secret_key)
    
    # Constant-time comparison to prevent timing attacks
    is_valid = hmac.compare_digest(stored_hash, computed_hash)
    
    # Find mismatched fields if invalid
    mismatch_fields = []
    if not is_valid:
        mismatch_fields = _find_mismatches(prediction_data)
    
    return PredictionIntegrity(
        prediction_id=prediction_data.get("id") or prediction_data.get("prediction_id"),
        original_hash=stored_hash,
        computed_hash=computed_hash,
        is_valid=is_valid,
        verified_at=datetime.utcnow(),
        mismatch_fields=mismatch_fields,
    )


def _create_canonical_data(
    prediction_data: Dict[str, Any],
    timestamp: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    Create canonical data structure for hashing.
    
    Extracts and normalizes the relevant fields for hashing
    to ensure consistent hash generation.
    """
    locked_at = timestamp or datetime.utcnow()
    
    # Extract fields that should be included in hash
    canonical = {
        "game_id": str(prediction_data.get("game_id", "")),
        "bet_type": str(prediction_data.get("bet_type", "")),
        "predicted_side": str(prediction_data.get("predicted_side", "")),
        "probability": round(float(prediction_data.get("probability", 0)), 6),
        "line_at_prediction": round(float(prediction_data.get("line_at_prediction") or prediction_data.get("line") or 0), 2),
        "odds_at_prediction": int(prediction_data.get("odds_at_prediction") or prediction_data.get("odds") or 0),
        "locked_at": locked_at.isoformat() if isinstance(locked_at, datetime) else str(locked_at),
    }
    
    # Include optional fields if present
    if "sport_code" in prediction_data:
        canonical["sport_code"] = str(prediction_data["sport_code"])
    
    if "signal_tier" in prediction_data:
        canonical["signal_tier"] = str(prediction_data["signal_tier"])
    
    if "model_id" in prediction_data:
        canonical["model_id"] = str(prediction_data["model_id"])
    
    return canonical


def _find_mismatches(prediction_data: Dict[str, Any]) -> list:
    """
    Identify fields that might have been modified.
    
    Note: This is a heuristic check since we can't know the original
    values without the original data.
    """
    potential_issues = []
    
    # Check for suspicious values
    prob = prediction_data.get("probability", 0)
    if prob < 0 or prob > 1:
        potential_issues.append("probability_out_of_range")
    
    odds = prediction_data.get("odds_at_prediction") or prediction_data.get("odds", 0)
    if odds != 0 and (odds < -10000 or odds > 10000):
        potential_issues.append("odds_suspicious")
    
    line = prediction_data.get("line_at_prediction") or prediction_data.get("line", 0)
    if abs(line) > 100:
        potential_issues.append("line_suspicious")
    
    return potential_issues


class PredictionHasher:
    """
    Utility class for managing prediction hashing.
    
    Provides methods for generating and verifying prediction
    integrity hashes with configurable options.
    """
    
    def __init__(
        self,
        secret_key: Optional[str] = None,
        include_metadata: bool = False,
    ):
        """
        Initialize hasher.
        
        Args:
            secret_key: HMAC secret for keyed hashing
            include_metadata: Whether to include extra metadata in hash
        """
        self.secret_key = secret_key
        self.include_metadata = include_metadata
    
    def hash(
        self,
        prediction_data: Dict[str, Any],
        timestamp: Optional[datetime] = None,
    ) -> str:
        """
        Generate hash for prediction.
        
        Args:
            prediction_data: Prediction data dictionary
            timestamp: Lock-in timestamp
            
        Returns:
            SHA-256 hash string
        """
        return hash_prediction(
            prediction_data,
            timestamp,
            self.secret_key,
        )
    
    def verify(
        self,
        prediction_data: Dict[str, Any],
        stored_hash: str,
        timestamp: datetime,
    ) -> PredictionIntegrity:
        """
        Verify prediction integrity.
        
        Args:
            prediction_data: Prediction data dictionary
            stored_hash: Original hash
            timestamp: Original timestamp
            
        Returns:
            PredictionIntegrity result
        """
        return verify_prediction_hash(
            prediction_data,
            stored_hash,
            timestamp,
            self.secret_key,
        )
    
    def batch_hash(
        self,
        predictions: list,
    ) -> Dict[str, str]:
        """
        Generate hashes for multiple predictions.
        
        Args:
            predictions: List of prediction dictionaries
            
        Returns:
            Dictionary mapping prediction IDs to hashes
        """
        hashes = {}
        
        for pred in predictions:
            pred_id = str(pred.get("id") or pred.get("prediction_id", ""))
            if pred_id:
                hashes[pred_id] = self.hash(pred)
        
        return hashes
    
    def batch_verify(
        self,
        predictions: list,
    ) -> Dict[str, PredictionIntegrity]:
        """
        Verify multiple predictions.
        
        Args:
            predictions: List of prediction dictionaries with stored hashes
            
        Returns:
            Dictionary mapping prediction IDs to integrity results
        """
        results = {}
        
        for pred in predictions:
            pred_id = str(pred.get("id") or pred.get("prediction_id", ""))
            stored_hash = pred.get("integrity_hash") or pred.get("hash")
            timestamp = pred.get("locked_at") or pred.get("created_at")
            
            if pred_id and stored_hash and timestamp:
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                
                results[pred_id] = self.verify(pred, stored_hash, timestamp)
        
        return results


def generate_prediction_receipt(
    prediction_data: Dict[str, Any],
    hash_value: str,
) -> Dict[str, Any]:
    """
    Generate a verification receipt for a prediction.
    
    Creates a human-readable receipt that can be used to
    verify the prediction was locked in at a specific time.
    
    Args:
        prediction_data: Prediction data
        hash_value: SHA-256 hash
        
    Returns:
        Receipt dictionary
    """
    return {
        "receipt_type": "prediction_lock_in",
        "version": "1.0",
        "prediction_id": str(prediction_data.get("id") or prediction_data.get("prediction_id", "")),
        "game_id": str(prediction_data.get("game_id", "")),
        "bet_type": prediction_data.get("bet_type"),
        "predicted_side": prediction_data.get("predicted_side"),
        "probability": round(float(prediction_data.get("probability", 0)), 4),
        "line": prediction_data.get("line_at_prediction") or prediction_data.get("line"),
        "odds": prediction_data.get("odds_at_prediction") or prediction_data.get("odds"),
        "locked_at": datetime.utcnow().isoformat(),
        "integrity_hash": hash_value,
        "hash_algorithm": "SHA-256",
        "verification_url": f"/api/v1/predictions/{prediction_data.get('id')}/verify",
    }


async def verify_prediction_in_db(
    db,
    prediction_id: UUID,
    secret_key: Optional[str] = None,
) -> Optional[PredictionIntegrity]:
    """
    Verify a prediction's integrity from database.
    
    Args:
        db: Database session
        prediction_id: Prediction ID
        secret_key: HMAC secret
        
    Returns:
        PredictionIntegrity or None if not found
    """
    from sqlalchemy import select
    from app.models.models import Prediction
    
    result = await db.execute(
        select(Prediction).where(Prediction.id == prediction_id)
    )
    prediction = result.scalar_one_or_none()
    
    if not prediction or not prediction.integrity_hash:
        return None
    
    prediction_data = {
        "id": prediction.id,
        "game_id": prediction.game_id,
        "bet_type": prediction.bet_type,
        "predicted_side": prediction.predicted_side,
        "probability": prediction.probability,
        "line_at_prediction": prediction.line_at_prediction,
        "odds_at_prediction": prediction.odds_at_prediction,
    }
    
    return verify_prediction_hash(
        prediction_data,
        prediction.integrity_hash,
        prediction.locked_at or prediction.created_at,
        secret_key,
    )
