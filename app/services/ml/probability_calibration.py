"""
Probability Calibration Module for ROYALEY

Implements multiple calibration methods to ensure accurate probability estimates
for Kelly Criterion calculations.

Methods:
- Isotonic Regression (default): Non-parametric, monotonically increasing
- Platt Scaling: Logistic regression on predictions
- Temperature Scaling: Single parameter optimization
"""

import numpy as np
import pickle
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any
from pathlib import Path
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class CalibrationMetrics:
    """Metrics for evaluating calibration quality."""
    expected_calibration_error: float  # ECE - primary metric
    maximum_calibration_error: float   # MCE
    brier_score: float                 # Probability accuracy
    log_loss: float                    # Cross-entropy loss
    reliability_bins: List[Dict[str, float]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'expected_calibration_error': self.expected_calibration_error,
            'maximum_calibration_error': self.maximum_calibration_error,
            'brier_score': self.brier_score,
            'log_loss': self.log_loss,
            'reliability_bins': self.reliability_bins
        }


@dataclass
class CalibrationResult:
    """Result of calibration training."""
    method: str
    calibrator_path: str
    metrics_before: CalibrationMetrics
    metrics_after: CalibrationMetrics
    improvement: float  # ECE improvement percentage
    training_samples: int
    trained_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'method': self.method,
            'calibrator_path': self.calibrator_path,
            'metrics_before': self.metrics_before.to_dict(),
            'metrics_after': self.metrics_after.to_dict(),
            'improvement': self.improvement,
            'training_samples': self.training_samples,
            'trained_at': self.trained_at.isoformat()
        }


class IsotonicCalibrator:
    """
    Isotonic Regression Calibrator.
    
    Non-parametric approach that fits a monotonically increasing function.
    Best for well-ordered predictions.
    """
    
    def __init__(self):
        self.calibrator = None
        self.is_fitted = False
        
    def fit(self, probabilities: np.ndarray, labels: np.ndarray) -> 'IsotonicCalibrator':
        """
        Fit isotonic regression calibrator.
        
        Args:
            probabilities: Predicted probabilities (N,)
            labels: True binary labels (N,)
            
        Returns:
            Self for chaining
        """
        from sklearn.isotonic import IsotonicRegression
        
        # Ensure proper shapes
        probabilities = np.asarray(probabilities).ravel()
        labels = np.asarray(labels).ravel()
        
        # Clip probabilities to valid range
        probabilities = np.clip(probabilities, 0.001, 0.999)
        
        self.calibrator = IsotonicRegression(
            y_min=0.001,
            y_max=0.999,
            out_of_bounds='clip'
        )
        self.calibrator.fit(probabilities, labels)
        self.is_fitted = True
        
        logger.info(f"Isotonic calibrator fitted on {len(probabilities)} samples")
        return self
        
    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        """Transform probabilities using fitted calibrator."""
        if not self.is_fitted:
            raise ValueError("Calibrator not fitted. Call fit() first.")
            
        probabilities = np.asarray(probabilities).ravel()
        probabilities = np.clip(probabilities, 0.001, 0.999)
        
        calibrated = self.calibrator.predict(probabilities)
        return np.clip(calibrated, 0.001, 0.999)
        
    def save(self, path: str) -> str:
        """Save calibrator to file."""
        with open(path, 'wb') as f:
            pickle.dump(self.calibrator, f)
        return path
        
    @classmethod
    def load(cls, path: str) -> 'IsotonicCalibrator':
        """Load calibrator from file."""
        instance = cls()
        with open(path, 'rb') as f:
            instance.calibrator = pickle.load(f)
        instance.is_fitted = True
        return instance


class PlattCalibrator:
    """
    Platt Scaling Calibrator.
    
    Applies logistic regression to transform predictions using sigmoid.
    Works well with neural network outputs.
    """
    
    def __init__(self):
        self.a = 0.0  # Sigmoid slope
        self.b = 0.0  # Sigmoid intercept
        self.is_fitted = False
        
    def fit(self, probabilities: np.ndarray, labels: np.ndarray) -> 'PlattCalibrator':
        """
        Fit Platt scaling using logistic regression.
        
        Args:
            probabilities: Predicted probabilities (N,)
            labels: True binary labels (N,)
            
        Returns:
            Self for chaining
        """
        from sklearn.linear_model import LogisticRegression
        
        probabilities = np.asarray(probabilities).ravel()
        labels = np.asarray(labels).ravel()
        
        # Clip and transform to logits
        probabilities = np.clip(probabilities, 0.001, 0.999)
        logits = np.log(probabilities / (1 - probabilities))
        
        # Fit logistic regression
        lr = LogisticRegression(solver='lbfgs', max_iter=1000)
        lr.fit(logits.reshape(-1, 1), labels)
        
        self.a = lr.coef_[0][0]
        self.b = lr.intercept_[0]
        self.is_fitted = True
        
        logger.info(f"Platt calibrator fitted: a={self.a:.4f}, b={self.b:.4f}")
        return self
        
    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        """Transform probabilities using Platt scaling."""
        if not self.is_fitted:
            raise ValueError("Calibrator not fitted. Call fit() first.")
            
        probabilities = np.asarray(probabilities).ravel()
        probabilities = np.clip(probabilities, 0.001, 0.999)
        
        # Transform through sigmoid with learned parameters
        logits = np.log(probabilities / (1 - probabilities))
        scaled_logits = self.a * logits + self.b
        calibrated = 1 / (1 + np.exp(-scaled_logits))
        
        return np.clip(calibrated, 0.001, 0.999)
        
    def save(self, path: str) -> str:
        """Save calibrator parameters to file."""
        params = {'a': self.a, 'b': self.b}
        with open(path, 'wb') as f:
            pickle.dump(params, f)
        return path
        
    @classmethod
    def load(cls, path: str) -> 'PlattCalibrator':
        """Load calibrator from file."""
        instance = cls()
        with open(path, 'rb') as f:
            params = pickle.load(f)
        instance.a = params['a']
        instance.b = params['b']
        instance.is_fitted = True
        return instance


class TemperatureCalibrator:
    """
    Temperature Scaling Calibrator.
    
    Single parameter optimization that divides logits by learned temperature.
    Fast and effective for deep models.
    """
    
    def __init__(self):
        self.temperature = 1.0
        self.is_fitted = False
        
    def fit(self, probabilities: np.ndarray, labels: np.ndarray) -> 'TemperatureCalibrator':
        """
        Fit temperature scaling by optimizing negative log likelihood.
        
        Args:
            probabilities: Predicted probabilities (N,)
            labels: True binary labels (N,)
            
        Returns:
            Self for chaining
        """
        from scipy.optimize import minimize_scalar
        
        probabilities = np.asarray(probabilities).ravel()
        labels = np.asarray(labels).ravel()
        
        # Clip probabilities
        probabilities = np.clip(probabilities, 0.001, 0.999)
        logits = np.log(probabilities / (1 - probabilities))
        
        def nll_loss(temperature: float) -> float:
            """Negative log likelihood with temperature scaling."""
            if temperature <= 0:
                return float('inf')
            scaled_logits = logits / temperature
            scaled_probs = 1 / (1 + np.exp(-scaled_logits))
            scaled_probs = np.clip(scaled_probs, 1e-10, 1 - 1e-10)
            
            # Binary cross-entropy
            nll = -np.mean(labels * np.log(scaled_probs) + (1 - labels) * np.log(1 - scaled_probs))
            return nll
        
        # Optimize temperature
        result = minimize_scalar(nll_loss, bounds=(0.1, 10.0), method='bounded')
        self.temperature = result.x
        self.is_fitted = True
        
        logger.info(f"Temperature calibrator fitted: T={self.temperature:.4f}")
        return self
        
    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        """Transform probabilities using temperature scaling."""
        if not self.is_fitted:
            raise ValueError("Calibrator not fitted. Call fit() first.")
            
        probabilities = np.asarray(probabilities).ravel()
        probabilities = np.clip(probabilities, 0.001, 0.999)
        
        logits = np.log(probabilities / (1 - probabilities))
        scaled_logits = logits / self.temperature
        calibrated = 1 / (1 + np.exp(-scaled_logits))
        
        return np.clip(calibrated, 0.001, 0.999)
        
    def save(self, path: str) -> str:
        """Save temperature to file."""
        with open(path, 'wb') as f:
            pickle.dump({'temperature': self.temperature}, f)
        return path
        
    @classmethod
    def load(cls, path: str) -> 'TemperatureCalibrator':
        """Load calibrator from file."""
        instance = cls()
        with open(path, 'rb') as f:
            params = pickle.load(f)
        instance.temperature = params['temperature']
        instance.is_fitted = True
        return instance


class ProbabilityCalibrator:
    """
    Main probability calibration manager.
    
    Supports multiple calibration methods and automatic selection
    based on validation performance.
    """
    
    METHODS = {
        'isotonic': IsotonicCalibrator,
        'platt': PlattCalibrator,
        'temperature': TemperatureCalibrator
    }
    
    def __init__(self, method: str = 'isotonic', n_bins: int = 10):
        """
        Initialize calibrator.
        
        Args:
            method: Calibration method ('isotonic', 'platt', 'temperature')
            n_bins: Number of bins for calibration metrics
        """
        if method not in self.METHODS:
            raise ValueError(f"Unknown method: {method}. Use one of {list(self.METHODS.keys())}")
            
        self.method = method
        self.n_bins = n_bins
        self.calibrator = self.METHODS[method]()
        
    def calculate_metrics(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray
    ) -> CalibrationMetrics:
        """
        Calculate calibration metrics.
        
        Args:
            probabilities: Predicted probabilities
            labels: True binary labels
            
        Returns:
            CalibrationMetrics with ECE, MCE, Brier, log_loss
        """
        probabilities = np.asarray(probabilities).ravel()
        labels = np.asarray(labels).ravel()
        probabilities = np.clip(probabilities, 1e-10, 1 - 1e-10)
        
        # Expected Calibration Error (ECE)
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        ece = 0.0
        mce = 0.0
        reliability_bins = []
        
        for i in range(self.n_bins):
            in_bin = (probabilities > bin_boundaries[i]) & (probabilities <= bin_boundaries[i + 1])
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                avg_confidence = probabilities[in_bin].mean()
                avg_accuracy = labels[in_bin].mean()
                calibration_error = abs(avg_accuracy - avg_confidence)
                
                ece += prop_in_bin * calibration_error
                mce = max(mce, calibration_error)
                
                reliability_bins.append({
                    'bin': i,
                    'confidence': float(avg_confidence),
                    'accuracy': float(avg_accuracy),
                    'count': int(in_bin.sum()),
                    'error': float(calibration_error)
                })
        
        # Brier Score
        brier_score = np.mean((probabilities - labels) ** 2)
        
        # Log Loss
        log_loss = -np.mean(labels * np.log(probabilities) + (1 - labels) * np.log(1 - probabilities))
        
        return CalibrationMetrics(
            expected_calibration_error=float(ece),
            maximum_calibration_error=float(mce),
            brier_score=float(brier_score),
            log_loss=float(log_loss),
            reliability_bins=reliability_bins
        )
        
    def fit(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
        validation_split: float = 0.2
    ) -> CalibrationResult:
        """
        Fit calibrator and evaluate improvement.
        
        Args:
            probabilities: Predicted probabilities
            labels: True binary labels
            validation_split: Fraction for validation
            
        Returns:
            CalibrationResult with before/after metrics
        """
        probabilities = np.asarray(probabilities).ravel()
        labels = np.asarray(labels).ravel()
        
        # Split data
        n_samples = len(probabilities)
        n_val = int(n_samples * validation_split)
        
        # Shuffle indices
        indices = np.random.permutation(n_samples)
        train_idx = indices[n_val:]
        val_idx = indices[:n_val]
        
        train_probs = probabilities[train_idx]
        train_labels = labels[train_idx]
        val_probs = probabilities[val_idx]
        val_labels = labels[val_idx]
        
        # Calculate metrics before calibration
        metrics_before = self.calculate_metrics(val_probs, val_labels)
        
        # Fit calibrator
        self.calibrator.fit(train_probs, train_labels)
        
        # Calibrate validation set
        calibrated_probs = self.calibrator.predict(val_probs)
        
        # Calculate metrics after calibration
        metrics_after = self.calculate_metrics(calibrated_probs, val_labels)
        
        # Calculate improvement
        if metrics_before.expected_calibration_error > 0:
            improvement = (metrics_before.expected_calibration_error - metrics_after.expected_calibration_error) / metrics_before.expected_calibration_error * 100
        else:
            improvement = 0.0
            
        logger.info(
            f"Calibration {self.method}: ECE {metrics_before.expected_calibration_error:.4f} -> "
            f"{metrics_after.expected_calibration_error:.4f} ({improvement:.1f}% improvement)"
        )
        
        return CalibrationResult(
            method=self.method,
            calibrator_path='',  # Set when saved
            metrics_before=metrics_before,
            metrics_after=metrics_after,
            improvement=improvement,
            training_samples=len(train_probs)
        )
        
    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        """Transform probabilities using fitted calibrator."""
        return self.calibrator.predict(probabilities)
        
    def save(self, directory: str, name: str = 'calibrator') -> str:
        """Save calibrator to directory."""
        Path(directory).mkdir(parents=True, exist_ok=True)
        path = str(Path(directory) / f"{name}_{self.method}.pkl")
        self.calibrator.save(path)
        return path
        
    @classmethod
    def load(cls, path: str) -> 'ProbabilityCalibrator':
        """Load calibrator from file."""
        # Determine method from filename
        filename = Path(path).stem
        method = filename.split('_')[-1]
        
        if method not in cls.METHODS:
            # Default to isotonic
            method = 'isotonic'
            
        instance = cls(method=method)
        instance.calibrator = cls.METHODS[method].load(path)
        return instance


class AutoCalibrator:
    """
    Automatic calibrator that selects the best method based on validation performance.
    """
    
    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self.best_method = None
        self.best_calibrator = None
        self.results = {}
        
    def fit(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
        validation_split: float = 0.2
    ) -> Dict[str, CalibrationResult]:
        """
        Fit all calibration methods and select the best one.
        
        Args:
            probabilities: Predicted probabilities
            labels: True binary labels
            validation_split: Fraction for validation
            
        Returns:
            Dictionary of results for each method
        """
        methods = ['isotonic', 'platt', 'temperature']
        best_ece = float('inf')
        
        for method in methods:
            try:
                calibrator = ProbabilityCalibrator(method=method, n_bins=self.n_bins)
                result = calibrator.fit(probabilities, labels, validation_split)
                self.results[method] = result
                
                if result.metrics_after.expected_calibration_error < best_ece:
                    best_ece = result.metrics_after.expected_calibration_error
                    self.best_method = method
                    self.best_calibrator = calibrator
                    
            except Exception as e:
                logger.warning(f"Calibration method {method} failed: {e}")
                continue
                
        logger.info(f"Auto-calibration selected: {self.best_method} (ECE={best_ece:.4f})")
        return self.results
        
    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        """Transform using best calibrator."""
        if self.best_calibrator is None:
            raise ValueError("No calibrator fitted. Call fit() first.")
        return self.best_calibrator.predict(probabilities)
        
    def save(self, directory: str, name: str = 'auto_calibrator') -> str:
        """Save best calibrator."""
        if self.best_calibrator is None:
            raise ValueError("No calibrator fitted. Call fit() first.")
        return self.best_calibrator.save(directory, name)


def calibrate_predictions(
    probabilities: np.ndarray,
    labels: np.ndarray,
    method: str = 'auto',
    n_bins: int = 10
) -> Tuple[np.ndarray, CalibrationResult]:
    """
    Convenience function to calibrate predictions.
    
    Args:
        probabilities: Raw predicted probabilities
        labels: True binary labels
        method: 'auto', 'isotonic', 'platt', or 'temperature'
        n_bins: Number of bins for metrics
        
    Returns:
        Tuple of (calibrated_probabilities, calibration_result)
    """
    if method == 'auto':
        calibrator = AutoCalibrator(n_bins=n_bins)
        calibrator.fit(probabilities, labels)
        calibrated = calibrator.predict(probabilities)
        result = calibrator.results[calibrator.best_method]
    else:
        calibrator = ProbabilityCalibrator(method=method, n_bins=n_bins)
        result = calibrator.fit(probabilities, labels)
        calibrated = calibrator.predict(probabilities)
        
    return calibrated, result


# Example usage
if __name__ == '__main__':
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Create slightly miscalibrated predictions
    true_probs = np.random.beta(2, 5, n_samples)
    labels = (np.random.random(n_samples) < true_probs).astype(int)
    
    # Add calibration bias
    raw_probs = true_probs * 0.8 + 0.1  # Overconfident in middle range
    
    # Calibrate
    calibrated, result = calibrate_predictions(raw_probs, labels, method='isotonic')
    
    print(f"Method: {result.method}")
    print(f"ECE Before: {result.metrics_before.expected_calibration_error:.4f}")
    print(f"ECE After: {result.metrics_after.expected_calibration_error:.4f}")
    print(f"Improvement: {result.improvement:.1f}%")
