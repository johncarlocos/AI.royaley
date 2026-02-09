"""
Meta-Ensemble Module for ROYALEY

Combines predictions from multiple ML frameworks using optimized weights:
- H2O AutoML
- AutoGluon
- Sklearn Ensemble (XGBoost, LightGBM, CatBoost, Random Forest)
- TensorFlow/LSTM (Deep Learning)
- Quantum ML (PennyLane, Qiskit) [Optional]

Total: 19 algorithm components integrated into unified ensemble.
"""

import numpy as np
import pickle
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Tuple
from pathlib import Path
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class FrameworkPrediction:
    """Prediction from a single framework."""
    framework: str  # 'h2o', 'autogluon', 'sklearn', 'tensorflow', 'quantum'
    probability: float
    confidence: float
    model_id: str
    features_used: int


@dataclass
class ExtendedEnsembleWeights:
    """Weights for all frameworks in the extended ensemble."""
    h2o_weight: float = 0.25
    autogluon_weight: float = 0.25
    sklearn_weight: float = 0.25
    tensorflow_weight: float = 0.15
    quantum_weight: float = 0.10
    
    def __post_init__(self):
        """Normalize weights to sum to 1."""
        total = (self.h2o_weight + self.autogluon_weight + self.sklearn_weight + 
                 self.tensorflow_weight + self.quantum_weight)
        if total > 0:
            self.h2o_weight /= total
            self.autogluon_weight /= total
            self.sklearn_weight /= total
            self.tensorflow_weight /= total
            self.quantum_weight /= total
            
    def to_dict(self) -> Dict[str, float]:
        return {
            'h2o': self.h2o_weight,
            'autogluon': self.autogluon_weight,
            'sklearn': self.sklearn_weight,
            'tensorflow': self.tensorflow_weight,
            'quantum': self.quantum_weight,
        }
        
    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> 'ExtendedEnsembleWeights':
        return cls(
            h2o_weight=d.get('h2o', 0.25),
            autogluon_weight=d.get('autogluon', 0.25),
            sklearn_weight=d.get('sklearn', 0.25),
            tensorflow_weight=d.get('tensorflow', 0.15),
            quantum_weight=d.get('quantum', 0.10),
        )


@dataclass
class EnsembleWeights:
    """Weights for each framework in the ensemble."""
    h2o_weight: float = 0.33
    autogluon_weight: float = 0.34
    sklearn_weight: float = 0.33
    
    def __post_init__(self):
        """Normalize weights to sum to 1."""
        total = self.h2o_weight + self.autogluon_weight + self.sklearn_weight
        if total > 0:
            self.h2o_weight /= total
            self.autogluon_weight /= total
            self.sklearn_weight /= total
            
    def to_dict(self) -> Dict[str, float]:
        return {
            'h2o': self.h2o_weight,
            'autogluon': self.autogluon_weight,
            'sklearn': self.sklearn_weight
        }
        
    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> 'EnsembleWeights':
        return cls(
            h2o_weight=d.get('h2o', 0.33),
            autogluon_weight=d.get('autogluon', 0.34),
            sklearn_weight=d.get('sklearn', 0.33)
        )


@dataclass
class EnsemblePrediction:
    """Combined prediction from meta-ensemble."""
    probability: float
    confidence: float
    signal_tier: str  # A, B, C, D
    edge: float
    framework_predictions: Dict[str, float]
    weights_used: Dict[str, float]
    agreement_score: float  # How much frameworks agree
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'probability': self.probability,
            'confidence': self.confidence,
            'signal_tier': self.signal_tier,
            'edge': self.edge,
            'framework_predictions': self.framework_predictions,
            'weights_used': self.weights_used,
            'agreement_score': self.agreement_score
        }


@dataclass
class WeightOptimizationResult:
    """Result of weight optimization process."""
    optimal_weights: EnsembleWeights
    validation_accuracy: float
    validation_auc: float
    validation_log_loss: float
    improvement_over_equal: float  # % improvement over equal weights
    optimization_method: str
    iterations: int
    optimized_at: datetime = field(default_factory=datetime.utcnow)


class MetaEnsemble:
    """
    Meta-Ensemble combining multiple ML frameworks.
    
    Uses performance-based weight optimization to combine
    H2O AutoML, AutoGluon, and Sklearn ensemble predictions.
    """
    
    # Signal tier thresholds
    TIER_THRESHOLDS = {
        'A': 0.65,  # Elite predictions
        'B': 0.60,  # Strong value
        'C': 0.55,  # Moderate confidence
        'D': 0.00   # Track only
    }
    
    def __init__(
        self,
        weights: Optional[EnsembleWeights] = None,
        min_frameworks: int = 2,
        calibrate: bool = True
    ):
        """
        Initialize meta-ensemble.
        
        Args:
            weights: Initial framework weights (defaults to equal)
            min_frameworks: Minimum frameworks required for prediction
            calibrate: Whether to apply probability calibration
        """
        self.weights = weights or EnsembleWeights()
        self.min_frameworks = min_frameworks
        self.calibrate = calibrate
        self.calibrator = None
        self.performance_history: List[Dict[str, float]] = []
        
    def fit(
        self,
        predictions: Dict[str, np.ndarray],
        y_true: np.ndarray,
        sport_code: str = None,
        bet_type: str = None,
        method: str = 'grid_search',
        metric: str = 'accuracy',
    ):
        """
        Fit meta-ensemble weights from a dict of framework predictions.
        
        Fully dynamic — every framework that trained successfully gets its
        own independent weight slot. No more hardcoded 3-slot limitation.
        
        Args:
            predictions: Dict mapping framework name to prediction arrays
                         e.g. {'h2o': [...], 'sklearn': [...], 'autogluon': [...], 'deep_learning': [...]}
            y_true: True labels
            sport_code: Sport code (for logging)
            bet_type: Bet type (for logging)
            method: Optimization method ('grid_search' or 'scipy')
            metric: Optimization metric ('accuracy', 'auc', 'log_loss')
            
        Returns:
            Object with .weights (dict), .auc, .accuracy, .log_loss attributes
        """
        y_true = np.asarray(y_true).ravel()
        fw_names = sorted(predictions.keys())
        fw_preds = {k: np.asarray(v).ravel() for k, v in predictions.items()}
        
        logger.info(f"Meta-ensemble fit: {len(fw_names)} frameworks ({', '.join(fw_names)}) for {sport_code} {bet_type}")
        
        # Run dynamic optimization over ALL frameworks
        opt_result = self._dynamic_optimize(
            fw_names=fw_names,
            fw_preds=fw_preds,
            labels=y_true,
            method=method,
            metric=metric,
        )
        
        # Store weights as simple dict for serialization
        weight_dict = opt_result['weights']
        
        # Build combined prediction for final metrics
        from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
        
        combined = np.zeros(len(y_true))
        for fw in fw_names:
            combined += weight_dict[fw] * fw_preds[fw]
        combined = np.clip(combined, 0.001, 0.999)
        
        final_accuracy = accuracy_score(y_true, combined > 0.5)
        final_auc = roc_auc_score(y_true, combined)
        final_logloss = log_loss(y_true, combined)
        
        # Also store in self.weights for backward compat
        self.weights = EnsembleWeights(
            h2o_weight=weight_dict.get('h2o', 0.0),
            autogluon_weight=weight_dict.get('autogluon', 0.0),
            sklearn_weight=weight_dict.get('sklearn', 0.0),
        )
        
        # Log per-framework individual accuracy
        for fw in fw_names:
            fw_acc = accuracy_score(y_true, fw_preds[fw] > 0.5)
            fw_auc = roc_auc_score(y_true, fw_preds[fw])
            logger.info(f"  {fw}: weight={weight_dict[fw]:.2f}, accuracy={fw_acc:.4f}, AUC={fw_auc:.4f}")
        
        logger.info(f"Meta-ensemble optimized: weights={weight_dict}, "
                     f"AUC={final_auc:.4f}, Acc={final_accuracy:.4f}")
        
        # Return result object matching training_service expectations
        return type('MetaEnsembleFitResult', (), {
            'weights': weight_dict,
            'auc': final_auc,
            'accuracy': final_accuracy,
            'log_loss': final_logloss,
            'improvement': opt_result['improvement'],
            'method': opt_result['method'],
        })()
    
    def _dynamic_optimize(
        self,
        fw_names: List[str],
        fw_preds: Dict[str, np.ndarray],
        labels: np.ndarray,
        method: str = 'grid_search',
        metric: str = 'accuracy',
    ) -> Dict[str, Any]:
        """
        Dynamic weight optimization for any number of frameworks.
        
        Uses grid search for ≤4 frameworks, scipy for 5+.
        """
        from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
        
        n_fw = len(fw_names)
        
        # Baseline: equal weights
        equal_w = 1.0 / n_fw
        baseline = np.zeros(len(labels))
        for fw in fw_names:
            baseline += equal_w * fw_preds[fw]
        baseline_accuracy = accuracy_score(labels, baseline > 0.5)
        
        def score_weights(weights_dict):
            """Score a given weight combination."""
            combined = np.zeros(len(labels))
            for fw, w in weights_dict.items():
                combined += w * fw_preds[fw]
            combined = np.clip(combined, 0.001, 0.999)
            
            if metric == 'accuracy':
                return accuracy_score(labels, combined > 0.5)
            elif metric == 'auc':
                return roc_auc_score(labels, combined)
            elif metric == 'log_loss':
                return -log_loss(labels, combined)
            return accuracy_score(labels, combined > 0.5)
        
        # Grid search (feasible for ≤5 frameworks)
        if n_fw <= 5:
            step = 0.05
            best_score = -float('inf')
            best_weights = {fw: equal_w for fw in fw_names}
            iterations = 0
            
            # Generate all weight combos that sum to 1.0
            import itertools
            grid_vals = np.arange(0.0, 1.01, step)
            
            for combo in itertools.product(grid_vals, repeat=n_fw - 1):
                last_w = 1.0 - sum(combo)
                if last_w < -0.001 or last_w > 1.001:
                    continue
                last_w = max(0.0, min(1.0, last_w))
                
                all_weights = list(combo) + [last_w]
                weights_dict = {fw: w for fw, w in zip(fw_names, all_weights)}
                
                iterations += 1
                score = score_weights(weights_dict)
                
                if score > best_score:
                    best_score = score
                    best_weights = weights_dict.copy()
            
            improvement = (best_score - baseline_accuracy) / baseline_accuracy * 100 if baseline_accuracy > 0 else 0
            
            # Log results
            weight_str = ", ".join(f"{fw}={w:.2f}" for fw, w in best_weights.items())
            logger.info(f"Weight optimization complete ({iterations} combos): {weight_str}")
            logger.info(f"Accuracy: {baseline_accuracy:.4f} -> {best_score:.4f} ({improvement:.1f}% improvement)")
            
            return {
                'weights': best_weights,
                'improvement': improvement,
                'method': f'grid_search_{iterations}_combos',
            }
        
        else:
            # Scipy optimization for 6+ frameworks
            from scipy.optimize import minimize as scipy_minimize
            
            def objective(x):
                last_w = 1.0 - sum(x)
                if last_w < 0:
                    return 1e10
                weights_dict = {fw: w for fw, w in zip(fw_names[:-1], x)}
                weights_dict[fw_names[-1]] = last_w
                return -score_weights(weights_dict)
            
            x0 = [equal_w] * (n_fw - 1)
            bounds = [(0, 1)] * (n_fw - 1)
            
            result = scipy_minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
            
            opt_weights = list(result.x) + [max(0, 1.0 - sum(result.x))]
            total = sum(opt_weights)
            opt_weights = [w / total for w in opt_weights]
            
            best_weights = {fw: w for fw, w in zip(fw_names, opt_weights)}
            best_score = score_weights(best_weights)
            improvement = (best_score - baseline_accuracy) / baseline_accuracy * 100 if baseline_accuracy > 0 else 0
            
            return {
                'weights': best_weights,
                'improvement': improvement,
                'method': 'scipy_optimize',
            }

    def combine_predictions(
        self,
        h2o_prob: Optional[float] = None,
        autogluon_prob: Optional[float] = None,
        sklearn_prob: Optional[float] = None,
        market_odds: Optional[float] = None
    ) -> EnsemblePrediction:
        """
        Combine predictions from multiple frameworks.
        
        Args:
            h2o_prob: H2O AutoML probability
            autogluon_prob: AutoGluon probability
            sklearn_prob: Sklearn ensemble probability
            market_odds: Market implied probability (for edge calculation)
            
        Returns:
            EnsemblePrediction with combined probability and metadata
        """
        predictions = {}
        weights = {}
        
        # Collect available predictions
        if h2o_prob is not None and 0 < h2o_prob < 1:
            predictions['h2o'] = h2o_prob
            weights['h2o'] = self.weights.h2o_weight
            
        if autogluon_prob is not None and 0 < autogluon_prob < 1:
            predictions['autogluon'] = autogluon_prob
            weights['autogluon'] = self.weights.autogluon_weight
            
        if sklearn_prob is not None and 0 < sklearn_prob < 1:
            predictions['sklearn'] = sklearn_prob
            weights['sklearn'] = self.weights.sklearn_weight
            
        # Check minimum frameworks
        if len(predictions) < self.min_frameworks:
            logger.warning(
                f"Only {len(predictions)} frameworks available, "
                f"minimum {self.min_frameworks} required"
            )
            # Proceed with available predictions
            
        if not predictions:
            raise ValueError("No valid predictions available")
            
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
            
        # Calculate weighted average
        combined_prob = sum(predictions[k] * weights[k] for k in predictions)
        
        # Apply calibration if available
        if self.calibrate and self.calibrator is not None:
            try:
                combined_prob = self.calibrator.predict(np.array([combined_prob]))[0]
            except Exception as e:
                logger.warning(f"Calibration failed: {e}")
                
        # Clip probability
        combined_prob = np.clip(combined_prob, 0.001, 0.999)
        
        # Calculate agreement score (how much frameworks agree)
        if len(predictions) > 1:
            probs = list(predictions.values())
            agreement = 1 - np.std(probs) / 0.5  # Normalize by max possible std
            agreement = max(0, min(1, agreement))
        else:
            agreement = 0.0
            
        # Determine signal tier
        signal_tier = self._get_signal_tier(combined_prob)
        
        # Calculate edge
        if market_odds is not None and market_odds > 0:
            edge = combined_prob - market_odds
        else:
            edge = combined_prob - 0.5  # Default edge vs coin flip
            
        # Calculate confidence (combination of probability and agreement)
        confidence = combined_prob * (0.7 + 0.3 * agreement)
        
        return EnsemblePrediction(
            probability=float(combined_prob),
            confidence=float(confidence),
            signal_tier=signal_tier,
            edge=float(edge),
            framework_predictions=predictions,
            weights_used=weights,
            agreement_score=float(agreement)
        )
        
    def _get_signal_tier(self, probability: float) -> str:
        """Determine signal tier based on probability."""
        # Use the higher of probability or 1-probability (for both sides)
        confidence = max(probability, 1 - probability)
        
        if confidence >= self.TIER_THRESHOLDS['A']:
            return 'A'
        elif confidence >= self.TIER_THRESHOLDS['B']:
            return 'B'
        elif confidence >= self.TIER_THRESHOLDS['C']:
            return 'C'
        else:
            return 'D'
            
    def optimize_weights(
        self,
        h2o_predictions: np.ndarray,
        autogluon_predictions: np.ndarray,
        sklearn_predictions: np.ndarray,
        labels: np.ndarray,
        method: str = 'grid_search',
        metric: str = 'accuracy'
    ) -> WeightOptimizationResult:
        """
        Optimize framework weights based on validation data.
        
        Args:
            h2o_predictions: H2O predicted probabilities
            autogluon_predictions: AutoGluon predicted probabilities
            sklearn_predictions: Sklearn predicted probabilities
            labels: True binary labels
            method: 'grid_search', 'bayesian', or 'gradient'
            metric: 'accuracy', 'auc', or 'log_loss'
            
        Returns:
            WeightOptimizationResult with optimal weights
        """
        h2o_predictions = np.asarray(h2o_predictions).ravel()
        autogluon_predictions = np.asarray(autogluon_predictions).ravel()
        sklearn_predictions = np.asarray(sklearn_predictions).ravel()
        labels = np.asarray(labels).ravel()
        
        # Calculate baseline with equal weights
        baseline_preds = (h2o_predictions + autogluon_predictions + sklearn_predictions) / 3
        baseline_accuracy = np.mean((baseline_preds > 0.5) == labels)
        
        if method == 'grid_search':
            return self._grid_search_optimization(
                h2o_predictions, autogluon_predictions, sklearn_predictions,
                labels, metric, baseline_accuracy
            )
        elif method == 'bayesian':
            return self._bayesian_optimization(
                h2o_predictions, autogluon_predictions, sklearn_predictions,
                labels, metric, baseline_accuracy
            )
        else:
            # Default to grid search
            return self._grid_search_optimization(
                h2o_predictions, autogluon_predictions, sklearn_predictions,
                labels, metric, baseline_accuracy
            )
            
    def _grid_search_optimization(
        self,
        h2o_preds: np.ndarray,
        ag_preds: np.ndarray,
        sk_preds: np.ndarray,
        labels: np.ndarray,
        metric: str,
        baseline_accuracy: float
    ) -> WeightOptimizationResult:
        """Grid search weight optimization."""
        from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
        
        best_score = -float('inf')
        best_weights = (0.33, 0.34, 0.33)
        iterations = 0
        
        # Grid search over weight combinations
        step = 0.05
        for w_h2o in np.arange(0.0, 1.01, step):
            for w_ag in np.arange(0.0, 1.01 - w_h2o, step):
                w_sk = 1.0 - w_h2o - w_ag
                
                if w_sk < 0:
                    continue
                    
                iterations += 1
                
                # Combine predictions
                combined = w_h2o * h2o_preds + w_ag * ag_preds + w_sk * sk_preds
                combined = np.clip(combined, 0.001, 0.999)
                
                # Calculate metric
                if metric == 'accuracy':
                    score = accuracy_score(labels, combined > 0.5)
                elif metric == 'auc':
                    score = roc_auc_score(labels, combined)
                elif metric == 'log_loss':
                    score = -log_loss(labels, combined)  # Negative because we maximize
                else:
                    score = accuracy_score(labels, combined > 0.5)
                    
                if score > best_score:
                    best_score = score
                    best_weights = (w_h2o, w_ag, w_sk)
                    
        # Set optimal weights
        self.weights = EnsembleWeights(
            h2o_weight=best_weights[0],
            autogluon_weight=best_weights[1],
            sklearn_weight=best_weights[2]
        )
        
        # Calculate final metrics
        final_preds = best_weights[0] * h2o_preds + best_weights[1] * ag_preds + best_weights[2] * sk_preds
        final_preds = np.clip(final_preds, 0.001, 0.999)
        
        final_accuracy = accuracy_score(labels, final_preds > 0.5)
        final_auc = roc_auc_score(labels, final_preds)
        final_log_loss = log_loss(labels, final_preds)
        
        improvement = (final_accuracy - baseline_accuracy) / baseline_accuracy * 100 if baseline_accuracy > 0 else 0
        
        logger.info(
            f"Weight optimization complete: H2O={best_weights[0]:.2f}, "
            f"AutoGluon={best_weights[1]:.2f}, Sklearn={best_weights[2]:.2f}"
        )
        logger.info(f"Accuracy: {baseline_accuracy:.4f} -> {final_accuracy:.4f} ({improvement:.1f}% improvement)")
        
        return WeightOptimizationResult(
            optimal_weights=self.weights,
            validation_accuracy=float(final_accuracy),
            validation_auc=float(final_auc),
            validation_log_loss=float(final_log_loss),
            improvement_over_equal=float(improvement),
            optimization_method='grid_search',
            iterations=iterations
        )
        
    def _bayesian_optimization(
        self,
        h2o_preds: np.ndarray,
        ag_preds: np.ndarray,
        sk_preds: np.ndarray,
        labels: np.ndarray,
        metric: str,
        baseline_accuracy: float
    ) -> WeightOptimizationResult:
        """Bayesian weight optimization using scipy."""
        from scipy.optimize import minimize
        from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
        
        iterations = [0]  # Use list to allow modification in nested function
        
        def objective(weights):
            """Objective function to minimize (negative score)."""
            iterations[0] += 1
            w_h2o, w_ag = weights
            w_sk = 1.0 - w_h2o - w_ag
            
            if w_sk < 0 or w_sk > 1:
                return 1e10
                
            combined = w_h2o * h2o_preds + w_ag * ag_preds + w_sk * sk_preds
            combined = np.clip(combined, 0.001, 0.999)
            
            if metric == 'accuracy':
                return -accuracy_score(labels, combined > 0.5)
            elif metric == 'auc':
                return -roc_auc_score(labels, combined)
            elif metric == 'log_loss':
                return log_loss(labels, combined)
            else:
                return -accuracy_score(labels, combined > 0.5)
                
        # Optimize
        from scipy.optimize import minimize
        
        result = minimize(
            objective,
            x0=[0.33, 0.34],  # Initial guess
            method='L-BFGS-B',
            bounds=[(0, 1), (0, 1)],
            options={'maxiter': 100}
        )
        
        w_h2o, w_ag = result.x
        w_sk = max(0, 1.0 - w_h2o - w_ag)
        
        # Normalize
        total = w_h2o + w_ag + w_sk
        w_h2o, w_ag, w_sk = w_h2o/total, w_ag/total, w_sk/total
        
        # Set optimal weights
        self.weights = EnsembleWeights(
            h2o_weight=w_h2o,
            autogluon_weight=w_ag,
            sklearn_weight=w_sk
        )
        
        # Calculate final metrics
        final_preds = w_h2o * h2o_preds + w_ag * ag_preds + w_sk * sk_preds
        final_preds = np.clip(final_preds, 0.001, 0.999)
        
        final_accuracy = accuracy_score(labels, final_preds > 0.5)
        final_auc = roc_auc_score(labels, final_preds)
        final_log_loss = log_loss(labels, final_preds)
        
        improvement = (final_accuracy - baseline_accuracy) / baseline_accuracy * 100 if baseline_accuracy > 0 else 0
        
        return WeightOptimizationResult(
            optimal_weights=self.weights,
            validation_accuracy=float(final_accuracy),
            validation_auc=float(final_auc),
            validation_log_loss=float(final_log_loss),
            improvement_over_equal=float(improvement),
            optimization_method='bayesian',
            iterations=iterations[0]
        )
        
    def update_weights_from_performance(
        self,
        framework_accuracies: Dict[str, float],
        learning_rate: float = 0.1
    ):
        """
        Update weights based on recent framework performance.
        
        Args:
            framework_accuracies: Dict of {framework: accuracy}
            learning_rate: How much to adjust weights
        """
        # Calculate performance-based weights
        total_accuracy = sum(framework_accuracies.values())
        if total_accuracy == 0:
            return
            
        target_weights = {
            k: v / total_accuracy
            for k, v in framework_accuracies.items()
        }
        
        # Exponential moving average update
        current_weights = self.weights.to_dict()
        new_weights = {}
        
        for framework in ['h2o', 'autogluon', 'sklearn']:
            current = current_weights.get(framework, 0.33)
            target = target_weights.get(framework, 0.33)
            new_weights[framework] = current + learning_rate * (target - current)
            
        # Update weights
        self.weights = EnsembleWeights.from_dict(new_weights)
        
        # Record history
        self.performance_history.append({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'accuracies': framework_accuracies,
            'weights': new_weights
        })
        
        logger.info(f"Weights updated: {new_weights}")
        
    def set_calibrator(self, calibrator: Any):
        """Set probability calibrator."""
        self.calibrator = calibrator
        
    def save(self, path: str) -> str:
        """Save ensemble configuration."""
        config = {
            'weights': self.weights.to_dict(),
            'min_frameworks': self.min_frameworks,
            'calibrate': self.calibrate,
            'tier_thresholds': self.TIER_THRESHOLDS,
            'performance_history': self.performance_history
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(config, f)
            
        return path
        
    @classmethod
    def load(cls, path: str) -> 'MetaEnsemble':
        """Load ensemble from file."""
        with open(path, 'rb') as f:
            config = pickle.load(f)
            
        weights = EnsembleWeights.from_dict(config['weights'])
        instance = cls(
            weights=weights,
            min_frameworks=config.get('min_frameworks', 2),
            calibrate=config.get('calibrate', True)
        )
        instance.performance_history = config.get('performance_history', [])
        
        return instance


class EnsemblePredictor:
    """
    High-level predictor using meta-ensemble with loaded models.
    """
    
    def __init__(
        self,
        h2o_model_path: Optional[str] = None,
        autogluon_model_path: Optional[str] = None,
        sklearn_model_path: Optional[str] = None,
        ensemble_config_path: Optional[str] = None
    ):
        """
        Initialize predictor with model paths.
        
        Args:
            h2o_model_path: Path to H2O MOJO
            autogluon_model_path: Path to AutoGluon model directory
            sklearn_model_path: Path to Sklearn pickle
            ensemble_config_path: Path to ensemble config
        """
        self.h2o_model = None
        self.autogluon_model = None
        self.sklearn_model = None
        
        # Load models
        if h2o_model_path:
            self._load_h2o_model(h2o_model_path)
        if autogluon_model_path:
            self._load_autogluon_model(autogluon_model_path)
        if sklearn_model_path:
            self._load_sklearn_model(sklearn_model_path)
            
        # Load or create ensemble
        if ensemble_config_path and Path(ensemble_config_path).exists():
            self.ensemble = MetaEnsemble.load(ensemble_config_path)
        else:
            self.ensemble = MetaEnsemble()
            
    def _load_h2o_model(self, path: str):
        """Load H2O MOJO model."""
        try:
            import h2o
            h2o.init()
            self.h2o_model = h2o.import_mojo(path)
            logger.info(f"H2O model loaded from {path}")
        except Exception as e:
            logger.warning(f"Failed to load H2O model: {e}")
            
    def _load_autogluon_model(self, path: str):
        """Load AutoGluon model."""
        try:
            from autogluon.tabular import TabularPredictor
            self.autogluon_model = TabularPredictor.load(path)
            logger.info(f"AutoGluon model loaded from {path}")
        except Exception as e:
            logger.warning(f"Failed to load AutoGluon model: {e}")
            
    def _load_sklearn_model(self, path: str):
        """Load Sklearn model."""
        try:
            with open(path, 'rb') as f:
                self.sklearn_model = pickle.load(f)
            logger.info(f"Sklearn model loaded from {path}")
        except Exception as e:
            logger.warning(f"Failed to load Sklearn model: {e}")
            
    def predict(
        self,
        features: Dict[str, float],
        market_odds: Optional[float] = None
    ) -> EnsemblePrediction:
        """
        Make prediction using all available models.
        
        Args:
            features: Feature dictionary
            market_odds: Market implied probability
            
        Returns:
            EnsemblePrediction with combined result
        """
        import pandas as pd
        
        feature_df = pd.DataFrame([features])
        
        h2o_prob = None
        ag_prob = None
        sk_prob = None
        
        # Get H2O prediction
        if self.h2o_model is not None:
            try:
                import h2o
                h2o_frame = h2o.H2OFrame(feature_df)
                pred = self.h2o_model.predict(h2o_frame)
                h2o_prob = pred['p1'].as_data_frame().iloc[0, 0]
            except Exception as e:
                logger.warning(f"H2O prediction failed: {e}")
                
        # Get AutoGluon prediction
        if self.autogluon_model is not None:
            try:
                pred = self.autogluon_model.predict_proba(feature_df)
                ag_prob = pred.iloc[0, 1]  # Probability of class 1
            except Exception as e:
                logger.warning(f"AutoGluon prediction failed: {e}")
                
        # Get Sklearn prediction
        if self.sklearn_model is not None:
            try:
                pred = self.sklearn_model.predict_proba(feature_df)
                sk_prob = pred[0, 1]
            except Exception as e:
                logger.warning(f"Sklearn prediction failed: {e}")
                
        # Combine predictions
        return self.ensemble.combine_predictions(
            h2o_prob=h2o_prob,
            autogluon_prob=ag_prob,
            sklearn_prob=sk_prob,
            market_odds=market_odds
        )
        
    def batch_predict(
        self,
        features_list: List[Dict[str, float]],
        market_odds_list: Optional[List[float]] = None
    ) -> List[EnsemblePrediction]:
        """
        Make predictions for multiple games.
        
        Args:
            features_list: List of feature dictionaries
            market_odds_list: List of market probabilities
            
        Returns:
            List of EnsemblePredictions
        """
        if market_odds_list is None:
            market_odds_list = [None] * len(features_list)
            
        predictions = []
        for features, odds in zip(features_list, market_odds_list):
            try:
                pred = self.predict(features, odds)
                predictions.append(pred)
            except Exception as e:
                logger.error(f"Prediction failed: {e}")
                # Return default prediction
                predictions.append(EnsemblePrediction(
                    probability=0.5,
                    confidence=0.0,
                    signal_tier='D',
                    edge=0.0,
                    framework_predictions={},
                    weights_used={},
                    agreement_score=0.0
                ))
                
        return predictions


# Example usage
if __name__ == '__main__':
    # Create sample predictions
    np.random.seed(42)
    n_samples = 1000
    
    # Simulated framework predictions
    true_probs = np.random.beta(2, 2, n_samples)
    labels = (np.random.random(n_samples) < true_probs).astype(int)
    
    h2o_preds = true_probs + np.random.normal(0, 0.1, n_samples)
    ag_preds = true_probs + np.random.normal(0, 0.08, n_samples)  # Better
    sk_preds = true_probs + np.random.normal(0, 0.12, n_samples)
    
    h2o_preds = np.clip(h2o_preds, 0.01, 0.99)
    ag_preds = np.clip(ag_preds, 0.01, 0.99)
    sk_preds = np.clip(sk_preds, 0.01, 0.99)
    
    # Optimize weights
    ensemble = MetaEnsemble()
    result = ensemble.optimize_weights(
        h2o_preds, ag_preds, sk_preds, labels,
        method='grid_search',
        metric='accuracy'
    )
    
    print(f"Optimal weights: {result.optimal_weights.to_dict()}")
    print(f"Accuracy: {result.validation_accuracy:.4f}")
    print(f"AUC: {result.validation_auc:.4f}")
    print(f"Improvement: {result.improvement_over_equal:.1f}%")
    
    # Make a single prediction
    pred = ensemble.combine_predictions(
        h2o_prob=0.65,
        autogluon_prob=0.68,
        sklearn_prob=0.62,
        market_odds=0.55
    )
    
    print(f"\nCombined prediction:")
    print(f"  Probability: {pred.probability:.4f}")
    print(f"  Signal Tier: {pred.signal_tier}")
    print(f"  Edge: {pred.edge:.4f}")
    print(f"  Agreement: {pred.agreement_score:.4f}")