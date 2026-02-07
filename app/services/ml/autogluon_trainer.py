"""
ROYALEY - AutoGluon Trainer
Phase 2: Multi-layer stack ensembling with AutoGluon

AutoGluon provides superior ensemble stacking that can improve
accuracy from 65% to 67-68% through:
- Multi-layer stacking architecture
- Automatic preprocessing and feature engineering
- Native GPU support for faster training
- Better probability calibration

INSTALLATION:
    pip install -r requirements-autogluon.txt
    # or
    pip install autogluon.tabular>=1.0.0
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import json
import numpy as np
import pandas as pd
import shutil

from .config import MLConfig, default_ml_config, BetType

logger = logging.getLogger(__name__)

# Check if AutoGluon is available
AUTOGLUON_AVAILABLE = False
try:
    from autogluon.tabular import TabularPredictor
    AUTOGLUON_AVAILABLE = True
except ImportError:
    logger.warning(
        "AutoGluon is not installed. Install with: pip install autogluon.tabular>=1.0.0\n"
        "AutoGluon training will not be available until installed."
    )


@dataclass
class AutoGluonModelResult:
    """Result from AutoGluon training"""
    model_id: str
    framework: str = "autogluon"
    sport_code: str = ""
    bet_type: str = ""
    
    # Best model info
    best_model: str = ""
    ensemble_size: int = 0
    
    # Performance metrics
    auc: float = 0.0
    accuracy: float = 0.0
    log_loss: float = 0.0
    balanced_accuracy: float = 0.0
    
    # Training info
    training_time_secs: float = 0.0
    n_training_samples: int = 0
    n_features: int = 0
    num_stack_levels: int = 0
    num_bag_folds: int = 0
    
    # Artifact path
    model_path: str = ""
    
    # Feature importance
    feature_importance: Dict[str, float] = field(default_factory=dict)
    
    # Leaderboard
    leaderboard: List[Dict] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)


class AutoGluonTrainer:
    """
    AutoGluon trainer for sports prediction models.
    
    Provides multi-layer stack ensembling with:
    - Automatic model selection
    - Hyperparameter tuning
    - GPU acceleration
    - Superior probability calibration
    """
    
    def __init__(
        self,
        config: MLConfig = None,
        model_dir: str = None,
    ):
        """
        Initialize AutoGluon trainer.
        
        Args:
            config: ML configuration
            model_dir: Directory for model artifacts
        """
        self.config = config or default_ml_config
        self.model_dir = Path(model_dir or self.config.model_artifact_path) / "autogluon"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self._predictor = None
    
    def train(
        self,
        train_df: pd.DataFrame,
        target_column: str,
        feature_columns: List[str],
        sport_code: str,
        bet_type: str,
        validation_df: pd.DataFrame = None,
        time_limit: int = None,
        presets: str = None,
        num_bag_folds: int = None,
        num_stack_levels: int = None,
    ) -> AutoGluonModelResult:
        """
        Train AutoGluon model with multi-layer stacking.
        
        Args:
            train_df: Training data
            target_column: Target variable column name
            feature_columns: Feature column names
            sport_code: Sport code (NFL, NBA, etc.)
            bet_type: Bet type (spread, moneyline, total)
            validation_df: Optional validation data
            time_limit: Training time limit in seconds
            presets: AutoGluon presets ('best_quality', 'high_quality', 'medium_quality')
            num_bag_folds: Number of bagging folds
            num_stack_levels: Number of stacking levels
            
        Returns:
            AutoGluonModelResult with trained model info
        """
        time_limit = time_limit or self.config.autogluon_time_limit
        presets = presets or self.config.autogluon_presets
        num_bag_folds = num_bag_folds or self.config.autogluon_num_bag_folds
        num_stack_levels = num_stack_levels or self.config.autogluon_num_stack_levels
        
        logger.info(
            f"Starting AutoGluon training for {sport_code} {bet_type} "
            f"with {len(train_df)} samples, presets={presets}"
        )
        
        start_time = datetime.now(timezone.utc)
        
        try:
            from autogluon.tabular import TabularPredictor
            
            # Prepare save path - FORCE clean to prevent "Learner is already fit"
            model_path = self.model_dir / sport_code / bet_type
            if model_path.exists():
                shutil.rmtree(model_path)
            # Do NOT mkdir here - let AutoGluon create it (avoids stale state)
            
            # Prepare training data
            train_data = train_df[feature_columns + [target_column]].copy()
            
            # Prepare validation data
            # IMPORTANT: best_quality preset uses bagging (num_bag_folds=8).
            # In bagged mode, tuning_data MUST be None - AutoGluon uses internal
            # cross-validation instead. Passing tuning_data causes DyStack to
            # partially fit the learner, then the main fit crashes with
            # "Learner is already fit."
            tuning_data = None
            use_bag_holdout = False
            if validation_df is not None:
                # Only use tuning_data with non-bagged presets
                if presets in ('medium_quality', 'good_quality'):
                    tuning_data = validation_df[feature_columns + [target_column]].copy()
                else:
                    # For best_quality: merge validation into training data
                    # AutoGluon will use internal CV for validation
                    extra_data = validation_df[feature_columns + [target_column]].copy()
                    train_data = pd.concat([train_data, extra_data], ignore_index=True)
                    logger.info(
                        f"Merged validation data into training set for bagged mode: "
                        f"{len(train_data)} total samples"
                    )
            
            # Create predictor
            predictor = TabularPredictor(
                label=target_column,
                problem_type='binary',
                eval_metric=self.config.autogluon_eval_metric,
                path=str(model_path),
            )
            
            # Configure hyperparameters for different algorithms
            hyperparameters = self._get_hyperparameters()
            
            # Train with multi-layer stacking
            predictor.fit(
                train_data=train_data,
                tuning_data=tuning_data,
                presets=presets,
                time_limit=time_limit,
                hyperparameters=hyperparameters,
                num_bag_folds=num_bag_folds,
                num_stack_levels=num_stack_levels,
                verbosity=2,
                # Excluded models:
                # - NN_TORCH, FASTAI: Too slow for production
                # - XGB: XGBoost 2.1+ removed n_classes_ attribute, incompatible
                #   with AutoGluon 1.5.0 predict_proba(). CatBoost/LightGBM
                #   perform identically. Re-enable after upgrading AutoGluon.
                excluded_model_types=['NN_TORCH', 'FASTAI', 'XGB'],
            )
            
            # Store predictor
            self._predictor = predictor
            
            # Get leaderboard
            leaderboard = predictor.leaderboard(silent=True)
            leaderboard_list = leaderboard.to_dict('records')
            
            # Get best model info (AutoGluon 1.5+ uses property, not method)
            best_model = predictor.model_best if hasattr(predictor, 'model_best') else 'WeightedEnsemble_L2'
            
            # Evaluate performance using OOF (out-of-fold) predictions
            # These are predictions on data the model DIDN'T see during training
            # This gives realistic, unbiased performance estimates
            
            # AUC from leaderboard (OOF-based)
            best_row = leaderboard[leaderboard['model'] == best_model]
            if not best_row.empty:
                auc = float(best_row['score_val'].iloc[0])
            else:
                auc = float(leaderboard['score_val'].iloc[0])
            
            # Accuracy from OOF predictions (realistic, not inflated)
            try:
                # get_oof_pred() returns OOF predictions for bagged models
                # These are predictions made on held-out folds during training
                oof_preds = predictor.get_oof_pred()
                y_true = train_data[target_column]
                # Align indices in case of any mismatch
                common_idx = oof_preds.index.intersection(y_true.index)
                accuracy = float((oof_preds.loc[common_idx] == y_true.loc[common_idx]).mean())
                logger.info(f"OOF accuracy: {accuracy:.4f} ({len(common_idx)} samples)")
            except Exception as e:
                logger.warning(f"Could not compute OOF accuracy: {e}, falling back to leaderboard")
                # Fallback: estimate from AUC (rough approximation)
                # For balanced binary classification, accuracy ≈ 0.5 + (AUC - 0.5) * 0.8
                accuracy = 0.5 + (auc - 0.5) * 0.8 if auc > 0.5 else 0.5
            
            # Get feature importance
            try:
                feat_imp = predictor.feature_importance(train_data)
                feat_imp_dict = dict(zip(
                    feat_imp.index.tolist(),
                    feat_imp['importance'].tolist()
                ))
            except:
                feat_imp_dict = {}
            
            training_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            # Create result
            result = AutoGluonModelResult(
                model_id=f"ag_{sport_code}_{bet_type}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
                sport_code=sport_code,
                bet_type=bet_type,
                best_model=best_model,
                ensemble_size=len(leaderboard),
                auc=auc if isinstance(auc, float) else 0.0,
                accuracy=accuracy if isinstance(accuracy, float) else 0.0,
                log_loss=0.0,
                balanced_accuracy=0.0,
                training_time_secs=training_time,
                n_training_samples=len(train_df),
                n_features=len(feature_columns),
                num_stack_levels=num_stack_levels,
                num_bag_folds=num_bag_folds,
                model_path=str(model_path),
                feature_importance=feat_imp_dict,
                leaderboard=leaderboard_list[:10],  # Top 10 models
            )
            
            logger.info(
                f"AutoGluon training complete: {result.model_id} "
                f"AUC={result.auc:.4f} Best={best_model}"
            )
            
            return result
            
        except ImportError:
            logger.error("AutoGluon not installed. Install with: pip install autogluon")
            raise
        except Exception as e:
            logger.error(f"AutoGluon training failed: {e}")
            raise
    
    def predict(
        self,
        model_path: str,
        data: pd.DataFrame,
        feature_columns: List[str],
    ) -> np.ndarray:
        """
        Make predictions using trained AutoGluon model.
        
        Args:
            model_path: Path to saved model
            data: Data to predict on
            feature_columns: Feature columns
            
        Returns:
            Array of predicted probabilities
        """
        try:
            from autogluon.tabular import TabularPredictor
            
            # Load predictor
            predictor = TabularPredictor.load(model_path)
            
            # Prepare data
            pred_data = data[feature_columns].copy()
            
            # Predict probabilities
            probs = predictor.predict_proba(pred_data)
            
            # Return probability of positive class
            if isinstance(probs, pd.DataFrame):
                # Get column for class 1
                if 1 in probs.columns:
                    return probs[1].values
                elif '1' in probs.columns:
                    return probs['1'].values
                else:
                    # Return second column (typically positive class)
                    return probs.iloc[:, 1].values
            else:
                return probs[:, 1]
                
        except Exception as e:
            logger.error(f"AutoGluon prediction failed: {e}")
            raise
    
    def predict_with_loaded(
        self,
        data: pd.DataFrame,
        feature_columns: List[str],
    ) -> np.ndarray:
        """
        Make predictions using already-loaded predictor.
        
        Args:
            data: Data to predict on
            feature_columns: Feature columns
            
        Returns:
            Array of predicted probabilities
        """
        if self._predictor is None:
            raise ValueError("No predictor loaded. Call train() or load() first.")
        
        pred_data = data[feature_columns].copy()
        probs = self._predictor.predict_proba(pred_data)
        
        if isinstance(probs, pd.DataFrame):
            if 1 in probs.columns:
                return probs[1].values
            elif '1' in probs.columns:
                return probs['1'].values
            else:
                return probs.iloc[:, 1].values
        else:
            return probs[:, 1]
    
    def load(self, model_path: str) -> None:
        """Load a saved AutoGluon model"""
        try:
            from autogluon.tabular import TabularPredictor
            self._predictor = TabularPredictor.load(model_path)
            logger.info(f"Loaded AutoGluon model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load AutoGluon model: {e}")
            raise
    
    def _get_hyperparameters(self) -> Dict:
        """Get hyperparameter configuration for different algorithms"""
        return {
            # Gradient Boosting
            'GBM': [
                {
                    'num_boost_round': 500,
                    'learning_rate': 0.05,
                    'num_leaves': 64,
                    'feature_fraction': 0.8,
                    'bagging_fraction': 0.8,
                },
                {
                    'num_boost_round': 300,
                    'learning_rate': 0.1,
                    'num_leaves': 128,
                },
            ],
            # XGBoost - DISABLED: xgboost 2.1+ incompatible with AutoGluon 1.5.0
            # Re-enable after upgrading AutoGluon or downgrading xgboost<2.1
            # 'XGB': [
            #     {
            #         'n_estimators': 500,
            #         'learning_rate': 0.05,
            #         'max_depth': 6,
            #         'subsample': 0.8,
            #         'colsample_bytree': 0.8,
            #     },
            # ],
            # CatBoost
            'CAT': [
                {
                    'iterations': 500,
                    'learning_rate': 0.05,
                    'depth': 6,
                },
            ],
            # Random Forest
            'RF': [
                {
                    'n_estimators': 300,
                    'max_depth': 12,
                    'min_samples_leaf': 5,
                },
            ],
            # Extra Trees
            'XT': [
                {
                    'n_estimators': 300,
                    'max_depth': 12,
                },
            ],
            # K-Nearest Neighbors
            'KNN': [
                {
                    'weights': 'distance',
                    'n_neighbors': 10,
                },
            ],
        }
    
    def get_leaderboard(
        self,
        sport_code: str,
        bet_type: str,
    ) -> Optional[pd.DataFrame]:
        """Get leaderboard for trained model"""
        if self._predictor is not None:
            return self._predictor.leaderboard(silent=True)
        
        model_path = self.model_dir / sport_code / bet_type
        if model_path.exists():
            try:
                from autogluon.tabular import TabularPredictor
                predictor = TabularPredictor.load(str(model_path))
                return predictor.leaderboard(silent=True)
            except:
                pass
        
        return None
    
    def get_model_info(self) -> Dict:
        """Get information about loaded model"""
        if self._predictor is None:
            return {}
        
        return {
            'best_model': self._predictor.model_best if hasattr(self._predictor, 'model_best') else 'unknown',
            'model_names': self._predictor.model_names() if callable(getattr(self._predictor, 'model_names', None)) else [],
            'problem_type': self._predictor.problem_type,
            'eval_metric': str(self._predictor.eval_metric),
        }


class AutoGluonTrainerMock:
    """
    Mock AutoGluon trainer for testing without AutoGluon installed.
    """
    
    def __init__(self, config: MLConfig = None, model_dir: str = None):
        self.config = config or default_ml_config
        self.model_dir = Path(model_dir or "./models/autogluon_mock")
        self.model_dir.mkdir(parents=True, exist_ok=True)
    
    def train(
        self,
        train_df: pd.DataFrame,
        target_column: str,
        feature_columns: List[str],
        sport_code: str,
        bet_type: str,
        **kwargs,
    ) -> AutoGluonModelResult:
        """Mock training that returns dummy results"""
        logger.info(f"Mock AutoGluon training for {sport_code} {bet_type}")
        
        # Simulate training time
        import time
        time.sleep(0.1)
        
        return AutoGluonModelResult(
            model_id=f"mock_ag_{sport_code}_{bet_type}",
            sport_code=sport_code,
            bet_type=bet_type,
            best_model="WeightedEnsemble_L2",
            ensemble_size=10,
            auc=0.64 + np.random.random() * 0.08,
            accuracy=0.60 + np.random.random() * 0.07,
            log_loss=0.62 + np.random.random() * 0.05,
            training_time_secs=120.0,
            n_training_samples=len(train_df),
            n_features=len(feature_columns),
            num_stack_levels=2,
            num_bag_folds=8,
            model_path=str(self.model_dir / f"{sport_code}_{bet_type}_mock"),
        )
    
    def predict(
        self,
        model_path: str,
        data: pd.DataFrame,
        feature_columns: List[str],
    ) -> np.ndarray:
        """Mock prediction"""
        n_samples = len(data)
        return np.random.beta(2.5, 2.5, n_samples)
    
    def load(self, model_path: str) -> None:
        """Mock load"""
        pass


def get_autogluon_trainer(
    config: MLConfig = None,
    use_mock: bool = False,
) -> AutoGluonTrainer:
    """
    Factory function to get AutoGluon trainer.
    
    Args:
        config: ML configuration
        use_mock: Use mock trainer for testing
        
    Returns:
        AutoGluon trainer instance
    """
    if use_mock:
        return AutoGluonTrainerMock(config)
    
    try:
        from autogluon.tabular import TabularPredictor
        return AutoGluonTrainer(config)
    except ImportError:
        logger.warning("AutoGluon not installed, using mock trainer")
        return AutoGluonTrainerMock(config)