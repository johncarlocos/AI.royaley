"""
LOYALEY - Walk-Forward Validation
Time-series aware validation to prevent data leakage
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from enum import Enum
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward validation"""
    # Time windows
    training_window_days: int = 365  # 1 year training
    test_window_days: int = 30  # 1 month test
    step_size_days: int = 30  # Monthly steps
    min_training_samples: int = 1000  # Minimum samples for training
    
    # Validation settings
    n_folds: Optional[int] = None  # If set, overrides step-based iteration
    gap_days: int = 1  # Gap between train and test to prevent leakage
    
    # Performance thresholds
    min_accuracy: float = 0.55
    min_auc: float = 0.58
    max_log_loss: float = 0.69  # -ln(0.5)
    
    # Parallel processing
    n_jobs: int = 4
    
    # Output options
    save_fold_models: bool = False
    calculate_feature_importance: bool = True


@dataclass
class FoldResult:
    """Results from a single validation fold"""
    fold_number: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    
    # Sample counts
    train_samples: int
    test_samples: int
    
    # Performance metrics
    accuracy: float
    auc: float
    log_loss: float
    brier_score: float
    
    # Betting metrics
    roi: float
    win_rate: float
    clv: float
    
    # Additional metrics
    precision: float
    recall: float
    f1_score: float
    
    # Feature importance (top 10)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    
    # Calibration
    calibration_error: float = 0.0
    
    # Predictions for analysis
    predictions: List[Dict] = field(default_factory=list)


@dataclass
class WalkForwardResult:
    """Complete walk-forward validation results"""
    config: WalkForwardConfig
    sport: str
    bet_type: str
    started_at: datetime
    completed_at: datetime
    
    # Overall metrics (averaged across folds)
    avg_accuracy: float
    std_accuracy: float
    avg_auc: float
    std_auc: float
    avg_log_loss: float
    avg_roi: float
    avg_win_rate: float
    avg_clv: float
    
    # Stability metrics
    accuracy_stability: float  # Lower is better (consistent performance)
    performance_trend: float  # Positive = improving over time
    
    # Pass/fail assessment
    passes_accuracy: bool
    passes_auc: bool
    passes_log_loss: bool
    overall_pass: bool
    
    # Fold details
    folds: List[FoldResult] = field(default_factory=list)
    
    # Feature importance (aggregated)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)


class WalkForwardValidator:
    """
    Walk-forward validation for time-series ML models
    Ensures no data leakage by maintaining strict temporal ordering
    """
    
    def __init__(
        self,
        model_trainer: Callable[[Any, Any], Any],
        model_predictor: Callable[[Any, Any], Tuple[np.ndarray, np.ndarray]],
        feature_generator: Optional[Callable[[Any], Any]] = None
    ):
        """
        Initialize walk-forward validator
        
        Args:
            model_trainer: Function to train model (X_train, y_train) -> model
            model_predictor: Function to make predictions (model, X_test) -> (predictions, probabilities)
            feature_generator: Optional function to generate features
        """
        self.model_trainer = model_trainer
        self.model_predictor = model_predictor
        self.feature_generator = feature_generator
        
    def validate(
        self,
        data: Any,
        config: WalkForwardConfig,
        sport: str,
        bet_type: str
    ) -> WalkForwardResult:
        """
        Run walk-forward validation
        
        Args:
            data: DataFrame or dataset with date column
            config: Validation configuration
            sport: Sport code (e.g., 'NBA')
            bet_type: Bet type (e.g., 'spread')
            
        Returns:
            WalkForwardResult with comprehensive metrics
        """
        started_at = datetime.now()
        logger.info(f"Starting walk-forward validation for {sport} {bet_type}")
        
        # Sort data by date
        data = self._sort_by_date(data)
        
        # Generate folds
        folds = self._generate_folds(data, config)
        logger.info(f"Generated {len(folds)} validation folds")
        
        # Run validation on each fold
        fold_results = []
        
        if config.n_jobs > 1:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=config.n_jobs) as executor:
                futures = {
                    executor.submit(self._validate_fold, data, fold, config, i): i
                    for i, fold in enumerate(folds)
                }
                
                for future in as_completed(futures):
                    fold_idx = futures[future]
                    try:
                        result = future.result()
                        fold_results.append(result)
                        logger.info(f"Fold {fold_idx + 1}/{len(folds)} complete: Acc={result.accuracy:.3f}, AUC={result.auc:.3f}")
                    except Exception as e:
                        logger.error(f"Fold {fold_idx} failed: {e}")
        else:
            # Sequential execution
            for i, fold in enumerate(folds):
                try:
                    result = self._validate_fold(data, fold, config, i)
                    fold_results.append(result)
                    logger.info(f"Fold {i + 1}/{len(folds)} complete: Acc={result.accuracy:.3f}, AUC={result.auc:.3f}")
                except Exception as e:
                    logger.error(f"Fold {i} failed: {e}")
        
        # Sort results by fold number
        fold_results.sort(key=lambda x: x.fold_number)
        
        # Calculate aggregate metrics
        completed_at = datetime.now()
        result = self._aggregate_results(fold_results, config, sport, bet_type, started_at, completed_at)
        
        logger.info(f"Walk-forward validation complete: Overall Pass={result.overall_pass}")
        return result
    
    def _sort_by_date(self, data: Any) -> Any:
        """Sort data by date column"""
        if hasattr(data, 'sort_values'):
            return data.sort_values('date')
        return data
    
    def _generate_folds(self, data: Any, config: WalkForwardConfig) -> List[Dict]:
        """Generate train/test fold specifications"""
        folds = []
        
        # Get date range
        if hasattr(data, 'iloc'):
            min_date = data['date'].min()
            max_date = data['date'].max()
        else:
            min_date = min(d['date'] for d in data)
            max_date = max(d['date'] for d in data)
        
        if isinstance(min_date, str):
            min_date = datetime.fromisoformat(min_date)
            max_date = datetime.fromisoformat(max_date)
        
        # Calculate number of folds
        if config.n_folds:
            total_days = (max_date - min_date).days
            step_days = (total_days - config.training_window_days - config.test_window_days) // (config.n_folds - 1)
            step_days = max(step_days, 1)
        else:
            step_days = config.step_size_days
        
        # Generate folds
        train_start = min_date
        
        while True:
            train_end = train_start + timedelta(days=config.training_window_days)
            test_start = train_end + timedelta(days=config.gap_days)
            test_end = test_start + timedelta(days=config.test_window_days)
            
            if test_end > max_date:
                break
            
            folds.append({
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end
            })
            
            train_start += timedelta(days=step_days)
            
            if config.n_folds and len(folds) >= config.n_folds:
                break
        
        return folds
    
    def _validate_fold(
        self,
        data: Any,
        fold: Dict,
        config: WalkForwardConfig,
        fold_number: int
    ) -> FoldResult:
        """Validate single fold"""
        
        # Split data
        train_data, test_data = self._split_data(data, fold)
        
        # Check minimum samples
        train_samples = len(train_data) if hasattr(train_data, '__len__') else train_data.shape[0]
        test_samples = len(test_data) if hasattr(test_data, '__len__') else test_data.shape[0]
        
        if train_samples < config.min_training_samples:
            logger.warning(f"Fold {fold_number}: Insufficient training samples ({train_samples})")
        
        # Generate features if needed
        if self.feature_generator:
            X_train, y_train = self.feature_generator(train_data)
            X_test, y_test = self.feature_generator(test_data)
        else:
            X_train = train_data.drop(columns=['target', 'date'], errors='ignore')
            y_train = train_data['target']
            X_test = test_data.drop(columns=['target', 'date'], errors='ignore')
            y_test = test_data['target']
        
        # Train model
        model = self.model_trainer(X_train, y_train)
        
        # Make predictions
        y_pred, y_prob = self.model_predictor(model, X_test)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_prob)
        
        # Calculate feature importance
        feature_importance = {}
        if config.calculate_feature_importance:
            feature_importance = self._get_feature_importance(model, X_train)
        
        # Calculate betting metrics
        betting_metrics = self._calculate_betting_metrics(test_data, y_pred, y_prob)
        
        return FoldResult(
            fold_number=fold_number,
            train_start=fold['train_start'],
            train_end=fold['train_end'],
            test_start=fold['test_start'],
            test_end=fold['test_end'],
            train_samples=train_samples,
            test_samples=test_samples,
            accuracy=metrics['accuracy'],
            auc=metrics['auc'],
            log_loss=metrics['log_loss'],
            brier_score=metrics['brier_score'],
            roi=betting_metrics['roi'],
            win_rate=betting_metrics['win_rate'],
            clv=betting_metrics['clv'],
            precision=metrics['precision'],
            recall=metrics['recall'],
            f1_score=metrics['f1'],
            feature_importance=feature_importance,
            calibration_error=metrics['calibration_error']
        )
    
    def _split_data(self, data: Any, fold: Dict) -> Tuple[Any, Any]:
        """Split data into train and test sets"""
        if hasattr(data, 'query'):
            # DataFrame
            train_mask = (data['date'] >= fold['train_start']) & (data['date'] < fold['train_end'])
            test_mask = (data['date'] >= fold['test_start']) & (data['date'] < fold['test_end'])
            return data[train_mask].copy(), data[test_mask].copy()
        else:
            # List of dicts
            train_data = [d for d in data if fold['train_start'] <= d['date'] < fold['train_end']]
            test_data = [d for d in data if fold['test_start'] <= d['date'] < fold['test_end']]
            return train_data, test_data
    
    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray
    ) -> Dict[str, float]:
        """Calculate classification metrics"""
        from sklearn.metrics import (
            accuracy_score, roc_auc_score, log_loss, brier_score_loss,
            precision_score, recall_score, f1_score
        )
        
        try:
            # Ensure arrays
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            y_prob = np.array(y_prob)
            
            # Handle probability shape
            if len(y_prob.shape) > 1:
                y_prob = y_prob[:, 1]  # Get positive class probability
            
            accuracy = accuracy_score(y_true, y_pred)
            
            try:
                auc = roc_auc_score(y_true, y_prob)
            except:
                auc = 0.5
            
            try:
                ll = log_loss(y_true, y_prob)
            except:
                ll = 1.0
            
            brier = brier_score_loss(y_true, y_prob)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            # Calibration error (simplified)
            calibration_error = self._calculate_calibration_error(y_true, y_prob)
            
            return {
                'accuracy': accuracy,
                'auc': auc,
                'log_loss': ll,
                'brier_score': brier,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'calibration_error': calibration_error
            }
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return {
                'accuracy': 0.5, 'auc': 0.5, 'log_loss': 1.0,
                'brier_score': 0.25, 'precision': 0.5, 'recall': 0.5,
                'f1': 0.5, 'calibration_error': 0.1
            }
    
    def _calculate_calibration_error(self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error"""
        try:
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            ece = 0.0
            
            for i in range(n_bins):
                mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
                if mask.sum() > 0:
                    bin_accuracy = y_true[mask].mean()
                    bin_confidence = y_prob[mask].mean()
                    bin_weight = mask.sum() / len(y_true)
                    ece += bin_weight * abs(bin_accuracy - bin_confidence)
            
            return ece
        except:
            return 0.1
    
    def _get_feature_importance(self, model: Any, X: Any) -> Dict[str, float]:
        """Extract feature importance from model"""
        try:
            # Try different attribute names
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_).flatten()
            else:
                return {}
            
            # Get feature names
            if hasattr(X, 'columns'):
                feature_names = list(X.columns)
            else:
                feature_names = [f'feature_{i}' for i in range(len(importances))]
            
            # Create sorted dict (top 10)
            importance_dict = dict(zip(feature_names, importances))
            sorted_importance = dict(sorted(
                importance_dict.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10])
            
            return sorted_importance
        except:
            return {}
    
    def _calculate_betting_metrics(
        self,
        test_data: Any,
        y_pred: np.ndarray,
        y_prob: np.ndarray
    ) -> Dict[str, float]:
        """Calculate betting performance metrics"""
        try:
            # This would calculate actual betting ROI based on odds
            # Simplified implementation
            correct = np.array(y_pred) == np.array(test_data['target'] if hasattr(test_data, 'target') else [d['target'] for d in test_data])
            win_rate = correct.mean() * 100
            
            # Simplified ROI calculation
            # Assuming -110 odds (4.55% vig)
            wins = correct.sum()
            losses = (~correct).sum()
            roi = ((wins * 0.909) - losses) / (wins + losses) * 100 if (wins + losses) > 0 else 0
            
            # Placeholder CLV
            clv = 0.0
            
            return {
                'roi': roi,
                'win_rate': win_rate,
                'clv': clv
            }
        except Exception as e:
            logger.error(f"Error calculating betting metrics: {e}")
            return {'roi': 0, 'win_rate': 50, 'clv': 0}
    
    def _aggregate_results(
        self,
        fold_results: List[FoldResult],
        config: WalkForwardConfig,
        sport: str,
        bet_type: str,
        started_at: datetime,
        completed_at: datetime
    ) -> WalkForwardResult:
        """Aggregate fold results into overall assessment"""
        
        if not fold_results:
            return WalkForwardResult(
                config=config,
                sport=sport,
                bet_type=bet_type,
                started_at=started_at,
                completed_at=completed_at,
                avg_accuracy=0.5,
                std_accuracy=0,
                avg_auc=0.5,
                std_auc=0,
                avg_log_loss=1.0,
                avg_roi=0,
                avg_win_rate=50,
                avg_clv=0,
                accuracy_stability=1.0,
                performance_trend=0,
                passes_accuracy=False,
                passes_auc=False,
                passes_log_loss=False,
                overall_pass=False
            )
        
        # Calculate averages
        accuracies = [f.accuracy for f in fold_results]
        aucs = [f.auc for f in fold_results]
        log_losses = [f.log_loss for f in fold_results]
        rois = [f.roi for f in fold_results]
        win_rates = [f.win_rate for f in fold_results]
        clvs = [f.clv for f in fold_results]
        
        avg_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)
        avg_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        avg_log_loss = np.mean(log_losses)
        avg_roi = np.mean(rois)
        avg_win_rate = np.mean(win_rates)
        avg_clv = np.mean(clvs)
        
        # Stability (coefficient of variation)
        accuracy_stability = std_accuracy / avg_accuracy if avg_accuracy > 0 else 1.0
        
        # Performance trend (linear regression slope)
        if len(accuracies) > 1:
            x = np.arange(len(accuracies))
            slope, _ = np.polyfit(x, accuracies, 1)
            performance_trend = slope
        else:
            performance_trend = 0
        
        # Pass/fail assessment
        passes_accuracy = avg_accuracy >= config.min_accuracy
        passes_auc = avg_auc >= config.min_auc
        passes_log_loss = avg_log_loss <= config.max_log_loss
        overall_pass = passes_accuracy and passes_auc and passes_log_loss
        
        # Aggregate feature importance
        all_importance = {}
        for fold in fold_results:
            for feature, importance in fold.feature_importance.items():
                if feature not in all_importance:
                    all_importance[feature] = []
                all_importance[feature].append(importance)
        
        feature_importance = {
            k: np.mean(v) for k, v in all_importance.items()
        }
        feature_importance = dict(sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10])
        
        # Generate recommendations
        recommendations = []
        
        if not passes_accuracy:
            recommendations.append(f"Accuracy ({avg_accuracy:.1%}) below threshold ({config.min_accuracy:.1%}). Consider feature engineering or model tuning.")
        
        if not passes_auc:
            recommendations.append(f"AUC ({avg_auc:.3f}) below threshold ({config.min_auc:.3f}). Model has weak discrimination ability.")
        
        if not passes_log_loss:
            recommendations.append(f"Log loss ({avg_log_loss:.3f}) above threshold ({config.max_log_loss:.3f}). Probability calibration needed.")
        
        if accuracy_stability > 0.2:
            recommendations.append(f"High accuracy variance (CV={accuracy_stability:.2f}). Model may be unstable.")
        
        if performance_trend < -0.01:
            recommendations.append(f"Declining performance trend ({performance_trend:.4f}). Model may need retraining more frequently.")
        
        if avg_roi < 0:
            recommendations.append(f"Negative ROI ({avg_roi:.1f}%). Review betting strategy and edge thresholds.")
        
        return WalkForwardResult(
            config=config,
            sport=sport,
            bet_type=bet_type,
            started_at=started_at,
            completed_at=completed_at,
            avg_accuracy=avg_accuracy,
            std_accuracy=std_accuracy,
            avg_auc=avg_auc,
            std_auc=std_auc,
            avg_log_loss=avg_log_loss,
            avg_roi=avg_roi,
            avg_win_rate=avg_win_rate,
            avg_clv=avg_clv,
            accuracy_stability=accuracy_stability,
            performance_trend=performance_trend,
            passes_accuracy=passes_accuracy,
            passes_auc=passes_auc,
            passes_log_loss=passes_log_loss,
            overall_pass=overall_pass,
            folds=fold_results,
            feature_importance=feature_importance,
            recommendations=recommendations
        )


class PurgedKFold:
    """
    Purged K-Fold cross-validation for time series
    Ensures no data leakage by adding gap between train and test
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        purge_days: int = 7,
        embargo_days: int = 1
    ):
        self.n_splits = n_splits
        self.purge_days = purge_days
        self.embargo_days = embargo_days
    
    def split(self, X, y=None, groups=None, dates=None):
        """
        Generate indices to split data into training and test set.
        
        Args:
            X: Features
            y: Target (unused)
            groups: Groups (unused)
            dates: Date array for temporal ordering
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        if dates is not None:
            # Sort by date
            sorted_idx = np.argsort(dates)
            indices = indices[sorted_idx]
        
        fold_size = n_samples // self.n_splits
        
        for i in range(self.n_splits):
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < self.n_splits - 1 else n_samples
            
            test_indices = indices[test_start:test_end]
            
            # Training indices with purge
            train_indices = np.concatenate([
                indices[:max(0, test_start - self.purge_days)],
                indices[min(n_samples, test_end + self.embargo_days):]
            ])
            
            yield train_indices, test_indices
    
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


# =============================================================================
# EXPORT ALIASES FOR BACKWARD COMPATIBILITY
# =============================================================================

WalkForwardEngine = WalkForwardValidator
"""Alias for WalkForwardValidator for backward compatibility."""

__all__ = [
    # Main Classes
    "WalkForwardValidator",
    "WalkForwardEngine",  # Alias
    "PurgedKFold",
    # Data Classes
    "WalkForwardConfig",
    "FoldResult",
    "WalkForwardResult",
]
