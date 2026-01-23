"""
ROYALEY - Walk-Forward Validation Framework
Phase 2: Time-series aware validation to prevent data leakage

Walk-forward validation ensures that training data always precedes
test data, which is critical for sports betting predictions where
future information must never leak into the model.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Generator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd

from .config import MLConfig, default_ml_config

logger = logging.getLogger(__name__)


@dataclass
class ValidationFold:
    """Single fold in walk-forward validation"""
    fold_number: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    n_train_samples: int = 0
    n_test_samples: int = 0


@dataclass 
class ValidationMetrics:
    """Metrics from a single validation fold"""
    fold_number: int
    
    # Classification metrics
    accuracy: float = 0.0
    auc: float = 0.0
    log_loss: float = 0.0
    brier_score: float = 0.0
    f1_score: float = 0.0
    
    # Calibration metrics
    expected_calibration_error: float = 0.0
    max_calibration_error: float = 0.0
    
    # Betting metrics
    clv: float = 0.0
    roi: float = 0.0
    win_rate: float = 0.0
    
    # Sample info
    n_predictions: int = 0
    n_correct: int = 0
    
    # Date range
    test_start: datetime = None
    test_end: datetime = None


@dataclass
class WalkForwardResult:
    """Complete walk-forward validation result"""
    sport_code: str
    bet_type: str
    
    # Fold metrics
    fold_metrics: List[ValidationMetrics] = field(default_factory=list)
    
    # Aggregated metrics
    mean_accuracy: float = 0.0
    std_accuracy: float = 0.0
    mean_auc: float = 0.0
    std_auc: float = 0.0
    mean_log_loss: float = 0.0
    mean_brier_score: float = 0.0
    mean_clv: float = 0.0
    mean_roi: float = 0.0
    
    # Configuration
    n_folds: int = 0
    training_window_days: int = 0
    validation_window_days: int = 0
    
    # Stability metrics
    accuracy_trend: float = 0.0  # Slope of accuracy over time
    is_stable: bool = True  # No significant degradation
    
    # Date range
    start_date: datetime = None
    end_date: datetime = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'sport_code': self.sport_code,
            'bet_type': self.bet_type,
            'mean_accuracy': self.mean_accuracy,
            'std_accuracy': self.std_accuracy,
            'mean_auc': self.mean_auc,
            'std_auc': self.std_auc,
            'mean_log_loss': self.mean_log_loss,
            'mean_brier_score': self.mean_brier_score,
            'mean_clv': self.mean_clv,
            'mean_roi': self.mean_roi,
            'n_folds': self.n_folds,
            'is_stable': self.is_stable,
        }


class WalkForwardValidator:
    """
    Walk-forward validation for time-series data.
    
    Implements rolling window validation where:
    - Training window: Last N days
    - Validation window: Next M days
    - Step size: Move forward by S days
    
    This prevents any future data leakage and simulates
    real-world prediction scenarios.
    """
    
    def __init__(
        self,
        config: MLConfig = None,
        training_window_days: int = None,
        validation_window_days: int = None,
        step_size_days: int = None,
        min_training_days: int = None,
        gap_days: int = None,
    ):
        """
        Initialize walk-forward validator.
        
        Args:
            config: ML configuration
            training_window_days: Size of training window
            validation_window_days: Size of validation window
            step_size_days: Days to step forward between folds
            min_training_days: Minimum required training data
            gap_days: Gap between training and validation to prevent leakage
        """
        self.config = config or default_ml_config
        
        self.training_window_days = (
            training_window_days or self.config.training_window_days
        )
        self.validation_window_days = (
            validation_window_days or self.config.validation_window_days
        )
        self.step_size_days = (
            step_size_days or self.config.step_size_days
        )
        self.min_training_days = (
            min_training_days or self.config.min_training_size_days
        )
        self.gap_days = gap_days or self.config.gap_days
    
    def generate_folds(
        self,
        data: pd.DataFrame,
        date_column: str,
        start_date: datetime = None,
        end_date: datetime = None,
    ) -> Generator[Tuple[ValidationFold, pd.DataFrame, pd.DataFrame], None, None]:
        """
        Generate walk-forward validation folds.
        
        Args:
            data: DataFrame with historical data
            date_column: Name of date column
            start_date: Optional start date for validation
            end_date: Optional end date for validation
            
        Yields:
            Tuples of (fold_info, train_df, test_df)
        """
        # Convert date column if needed
        if not pd.api.types.is_datetime64_any_dtype(data[date_column]):
            data = data.copy()
            data[date_column] = pd.to_datetime(data[date_column])
        
        # Sort by date
        data = data.sort_values(date_column).reset_index(drop=True)
        
        # Determine date range
        data_start = data[date_column].min()
        data_end = data[date_column].max()
        
        if start_date is None:
            start_date = data_start + timedelta(days=self.min_training_days)
        if end_date is None:
            end_date = data_end
        
        # Ensure start date allows minimum training window
        min_start = data_start + timedelta(days=self.min_training_days)
        if start_date < min_start:
            start_date = min_start
        
        logger.info(
            f"Generating walk-forward folds from {start_date.date()} to {end_date.date()}"
        )
        
        fold_number = 0
        current_test_start = start_date
        
        while current_test_start < end_date:
            # Calculate date boundaries
            train_end = current_test_start - timedelta(days=self.gap_days)
            train_start = train_end - timedelta(days=self.training_window_days)
            test_start = current_test_start
            test_end = min(
                test_start + timedelta(days=self.validation_window_days),
                end_date
            )
            
            # Ensure we have enough training data
            if train_start < data_start:
                train_start = data_start
            
            # Check if we have enough training data
            training_days = (train_end - train_start).days
            if training_days < self.min_training_days:
                logger.warning(
                    f"Fold {fold_number}: Insufficient training data "
                    f"({training_days} days < {self.min_training_days} required)"
                )
                current_test_start += timedelta(days=self.step_size_days)
                continue
            
            # Split data
            train_mask = (
                (data[date_column] >= train_start) &
                (data[date_column] < train_end)
            )
            test_mask = (
                (data[date_column] >= test_start) &
                (data[date_column] < test_end)
            )
            
            train_df = data[train_mask].copy()
            test_df = data[test_mask].copy()
            
            # Skip if no test data
            if len(test_df) == 0:
                current_test_start += timedelta(days=self.step_size_days)
                continue
            
            fold = ValidationFold(
                fold_number=fold_number,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                n_train_samples=len(train_df),
                n_test_samples=len(test_df),
            )
            
            logger.debug(
                f"Fold {fold_number}: Train {train_start.date()}-{train_end.date()} "
                f"({len(train_df)} samples), Test {test_start.date()}-{test_end.date()} "
                f"({len(test_df)} samples)"
            )
            
            yield fold, train_df, test_df
            
            fold_number += 1
            current_test_start += timedelta(days=self.step_size_days)
    
    def validate(
        self,
        trainer,
        data: pd.DataFrame,
        target_column: str,
        feature_columns: List[str],
        date_column: str,
        sport_code: str,
        bet_type: str,
        start_date: datetime = None,
        end_date: datetime = None,
    ) -> WalkForwardResult:
        """
        Run complete walk-forward validation.
        
        Args:
            trainer: Trainer instance (H2O, AutoGluon, or Sklearn)
            data: Historical data
            target_column: Target variable column
            feature_columns: Feature columns
            date_column: Date column
            sport_code: Sport code
            bet_type: Bet type
            start_date: Optional start date
            end_date: Optional end date
            
        Returns:
            WalkForwardResult with all metrics
        """
        from sklearn.metrics import (
            accuracy_score, roc_auc_score, log_loss, brier_score_loss, f1_score
        )
        
        logger.info(
            f"Starting walk-forward validation for {sport_code} {bet_type}"
        )
        
        fold_metrics = []
        
        for fold, train_df, test_df in self.generate_folds(
            data, date_column, start_date, end_date
        ):
            try:
                # Train model on this fold
                logger.info(f"Training fold {fold.fold_number}...")
                
                # Train
                trainer.train(
                    train_df=train_df,
                    target_column=target_column,
                    feature_columns=feature_columns,
                    sport_code=sport_code,
                    bet_type=bet_type,
                )
                
                # Predict on test set
                if hasattr(trainer, 'predict_with_loaded'):
                    probs = trainer.predict_with_loaded(test_df, feature_columns)
                else:
                    probs = trainer.predict(
                        model_path=trainer._model_path if hasattr(trainer, '_model_path') else '',
                        data=test_df,
                        feature_columns=feature_columns,
                    )
                
                # Calculate metrics
                y_true = test_df[target_column].values
                y_pred = (probs >= 0.5).astype(int)
                
                metrics = ValidationMetrics(
                    fold_number=fold.fold_number,
                    accuracy=accuracy_score(y_true, y_pred),
                    auc=roc_auc_score(y_true, probs) if len(np.unique(y_true)) > 1 else 0.5,
                    log_loss=log_loss(y_true, probs),
                    brier_score=brier_score_loss(y_true, probs),
                    f1_score=f1_score(y_true, y_pred, zero_division=0),
                    expected_calibration_error=self._calculate_ece(y_true, probs),
                    n_predictions=len(y_true),
                    n_correct=int(sum(y_true == y_pred)),
                    test_start=fold.test_start,
                    test_end=fold.test_end,
                )
                
                # Calculate win rate
                metrics.win_rate = metrics.n_correct / metrics.n_predictions if metrics.n_predictions > 0 else 0
                
                fold_metrics.append(metrics)
                
                logger.info(
                    f"Fold {fold.fold_number}: Accuracy={metrics.accuracy:.4f}, "
                    f"AUC={metrics.auc:.4f}, LogLoss={metrics.log_loss:.4f}"
                )
                
            except Exception as e:
                logger.error(f"Fold {fold.fold_number} failed: {e}")
                continue
        
        if not fold_metrics:
            logger.error("No successful folds in walk-forward validation")
            return WalkForwardResult(
                sport_code=sport_code,
                bet_type=bet_type,
            )
        
        # Aggregate metrics
        result = self._aggregate_metrics(
            fold_metrics, sport_code, bet_type
        )
        
        logger.info(
            f"Walk-forward validation complete: "
            f"Mean Accuracy={result.mean_accuracy:.4f}±{result.std_accuracy:.4f}, "
            f"Mean AUC={result.mean_auc:.4f}±{result.std_auc:.4f}"
        )
        
        return result
    
    def _aggregate_metrics(
        self,
        fold_metrics: List[ValidationMetrics],
        sport_code: str,
        bet_type: str,
    ) -> WalkForwardResult:
        """Aggregate metrics across all folds"""
        accuracies = [m.accuracy for m in fold_metrics]
        aucs = [m.auc for m in fold_metrics]
        log_losses = [m.log_loss for m in fold_metrics]
        brier_scores = [m.brier_score for m in fold_metrics]
        clvs = [m.clv for m in fold_metrics]
        rois = [m.roi for m in fold_metrics]
        
        # Calculate trend in accuracy
        if len(accuracies) > 1:
            x = np.arange(len(accuracies))
            slope, _ = np.polyfit(x, accuracies, 1)
            accuracy_trend = slope
        else:
            accuracy_trend = 0.0
        
        # Check stability (no significant degradation)
        is_stable = (
            accuracy_trend > -0.001 and  # Not declining significantly
            np.std(accuracies) < 0.05    # Not too variable
        )
        
        return WalkForwardResult(
            sport_code=sport_code,
            bet_type=bet_type,
            fold_metrics=fold_metrics,
            mean_accuracy=np.mean(accuracies),
            std_accuracy=np.std(accuracies),
            mean_auc=np.mean(aucs),
            std_auc=np.std(aucs),
            mean_log_loss=np.mean(log_losses),
            mean_brier_score=np.mean(brier_scores),
            mean_clv=np.mean(clvs),
            mean_roi=np.mean(rois),
            n_folds=len(fold_metrics),
            training_window_days=self.training_window_days,
            validation_window_days=self.validation_window_days,
            accuracy_trend=accuracy_trend,
            is_stable=is_stable,
            start_date=fold_metrics[0].test_start if fold_metrics else None,
            end_date=fold_metrics[-1].test_end if fold_metrics else None,
        )
    
    def _calculate_ece(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """
        Calculate Expected Calibration Error (ECE).
        
        ECE measures how well predicted probabilities match
        actual observed frequencies.
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        for i in range(n_bins):
            in_bin = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
            prop_in_bin = np.mean(in_bin)
            
            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(y_true[in_bin])
                avg_confidence_in_bin = np.mean(y_prob[in_bin])
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece


class TimeSeriesSplitter:
    """
    Utility class for creating train/test splits with temporal ordering.
    
    Ensures all training data comes before test data.
    """
    
    @staticmethod
    def train_test_split(
        data: pd.DataFrame,
        date_column: str,
        test_size: float = 0.2,
        gap_days: int = 1,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/test with temporal ordering.
        
        Args:
            data: DataFrame to split
            date_column: Date column name
            test_size: Proportion for test set
            gap_days: Gap between train and test
            
        Returns:
            Tuple of (train_df, test_df)
        """
        # Sort by date
        data = data.sort_values(date_column).reset_index(drop=True)
        
        # Find split point
        n_samples = len(data)
        split_idx = int(n_samples * (1 - test_size))
        
        train_df = data.iloc[:split_idx].copy()
        test_df = data.iloc[split_idx:].copy()
        
        # Apply gap
        if gap_days > 0:
            train_end = train_df[date_column].max()
            test_start = train_end + timedelta(days=gap_days)
            test_df = test_df[test_df[date_column] >= test_start].copy()
        
        return train_df, test_df
    
    @staticmethod
    def expanding_window_split(
        data: pd.DataFrame,
        date_column: str,
        initial_train_size: float = 0.5,
        step_size: float = 0.1,
    ) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
        """
        Generate expanding window splits.
        
        Training window grows over time while test window stays fixed.
        
        Args:
            data: DataFrame to split
            date_column: Date column name
            initial_train_size: Initial training set proportion
            step_size: Size of each test window
            
        Yields:
            Tuples of (train_df, test_df)
        """
        data = data.sort_values(date_column).reset_index(drop=True)
        n_samples = len(data)
        
        train_end_idx = int(n_samples * initial_train_size)
        test_size = int(n_samples * step_size)
        
        while train_end_idx + test_size <= n_samples:
            train_df = data.iloc[:train_end_idx].copy()
            test_df = data.iloc[train_end_idx:train_end_idx + test_size].copy()
            
            yield train_df, test_df
            
            train_end_idx += test_size
