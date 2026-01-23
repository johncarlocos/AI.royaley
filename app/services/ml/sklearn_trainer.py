"""
ROYALEY - Sklearn Ensemble Trainer
Phase 2: Custom ensemble combining XGBoost, LightGBM, CatBoost, and Random Forest

This module provides a fallback training option and contributes to
the meta-ensemble for improved predictions.
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import (
    RandomForestClassifier,
    VotingClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    log_loss,
    brier_score_loss,
)

from .config import MLConfig, default_ml_config

logger = logging.getLogger(__name__)


@dataclass
class SklearnModelResult:
    """Result from Sklearn ensemble training"""
    model_id: str
    framework: str = "sklearn"
    sport_code: str = ""
    bet_type: str = ""
    
    # Ensemble info
    ensemble_type: str = "stacking"  # stacking, voting
    base_models: List[str] = field(default_factory=list)
    
    # Performance metrics
    auc: float = 0.0
    accuracy: float = 0.0
    log_loss: float = 0.0
    brier_score: float = 0.0
    
    # Cross-validation metrics
    cv_auc_mean: float = 0.0
    cv_auc_std: float = 0.0
    cv_accuracy_mean: float = 0.0
    cv_accuracy_std: float = 0.0
    
    # Training info
    training_time_secs: float = 0.0
    n_training_samples: int = 0
    n_features: int = 0
    
    # Artifact path
    model_path: str = ""
    scaler_path: str = ""
    
    # Feature importance (from tree models)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    hyperparameters: Dict = field(default_factory=dict)


class SklearnEnsembleTrainer:
    """
    Sklearn-based ensemble trainer.
    
    Combines multiple gradient boosting and tree-based models:
    - XGBoost
    - LightGBM
    - CatBoost
    - Random Forest
    
    Uses stacking with logistic regression as meta-learner.
    """
    
    def __init__(
        self,
        config: MLConfig = None,
        model_dir: str = None,
    ):
        """
        Initialize Sklearn ensemble trainer.
        
        Args:
            config: ML configuration
            model_dir: Directory for model artifacts
        """
        self.config = config or default_ml_config
        self.model_dir = Path(model_dir or self.config.model_artifact_path) / "sklearn"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self._model = None
        self._scaler = None
        self._feature_columns = None
    
    def train(
        self,
        train_df: pd.DataFrame,
        target_column: str,
        feature_columns: List[str],
        sport_code: str,
        bet_type: str,
        validation_df: pd.DataFrame = None,
        use_stacking: bool = True,
        cv_folds: int = 5,
    ) -> SklearnModelResult:
        """
        Train Sklearn ensemble model.
        
        Args:
            train_df: Training data
            target_column: Target variable column name
            feature_columns: Feature column names
            sport_code: Sport code
            bet_type: Bet type
            validation_df: Optional validation data
            use_stacking: Use stacking (True) or voting (False)
            cv_folds: Number of cross-validation folds
            
        Returns:
            SklearnModelResult with trained model info
        """
        logger.info(
            f"Starting Sklearn ensemble training for {sport_code} {bet_type} "
            f"with {len(train_df)} samples"
        )
        
        start_time = datetime.now(timezone.utc)
        
        # Prepare data
        X_train = train_df[feature_columns].copy()
        y_train = train_df[target_column].values
        
        # Handle missing values
        X_train = X_train.fillna(0)
        
        # Scale features
        self._scaler = StandardScaler()
        X_train_scaled = self._scaler.fit_transform(X_train)
        
        # Store feature columns
        self._feature_columns = feature_columns
        
        # Create base models
        base_models = self._create_base_models()
        
        # Create ensemble
        if use_stacking:
            self._model = self._create_stacking_ensemble(base_models)
            ensemble_type = "stacking"
        else:
            self._model = self._create_voting_ensemble(base_models)
            ensemble_type = "voting"
        
        # Train ensemble
        logger.info("Training ensemble model...")
        self._model.fit(X_train_scaled, y_train)
        
        # Cross-validation
        logger.info("Running cross-validation...")
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_auc_scores = cross_val_score(
            self._model, X_train_scaled, y_train,
            cv=cv, scoring='roc_auc'
        )
        cv_acc_scores = cross_val_score(
            self._model, X_train_scaled, y_train,
            cv=cv, scoring='accuracy'
        )
        
        # Evaluate on training data
        train_probs = self._model.predict_proba(X_train_scaled)[:, 1]
        train_preds = self._model.predict(X_train_scaled)
        
        train_auc = roc_auc_score(y_train, train_probs)
        train_acc = accuracy_score(y_train, train_preds)
        train_logloss = log_loss(y_train, train_probs)
        train_brier = brier_score_loss(y_train, train_probs)
        
        # Evaluate on validation data if provided
        if validation_df is not None:
            X_val = validation_df[feature_columns].fillna(0)
            y_val = validation_df[target_column].values
            X_val_scaled = self._scaler.transform(X_val)
            
            val_probs = self._model.predict_proba(X_val_scaled)[:, 1]
            val_preds = self._model.predict(X_val_scaled)
            
            val_auc = roc_auc_score(y_val, val_probs)
            val_acc = accuracy_score(y_val, val_preds)
            val_logloss = log_loss(y_val, val_probs)
            val_brier = brier_score_loss(y_val, val_probs)
        else:
            val_auc = train_auc
            val_acc = train_acc
            val_logloss = train_logloss
            val_brier = train_brier
        
        # Get feature importance
        feature_importance = self._get_feature_importance(feature_columns)
        
        # Save model
        model_path = self._save_model(sport_code, bet_type)
        scaler_path = self._save_scaler(sport_code, bet_type)
        
        training_time = (datetime.now(timezone.utc) - start_time).total_seconds()
        
        # Create result
        result = SklearnModelResult(
            model_id=f"sk_{sport_code}_{bet_type}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
            sport_code=sport_code,
            bet_type=bet_type,
            ensemble_type=ensemble_type,
            base_models=list(base_models.keys()),
            auc=val_auc,
            accuracy=val_acc,
            log_loss=val_logloss,
            brier_score=val_brier,
            cv_auc_mean=cv_auc_scores.mean(),
            cv_auc_std=cv_auc_scores.std(),
            cv_accuracy_mean=cv_acc_scores.mean(),
            cv_accuracy_std=cv_acc_scores.std(),
            training_time_secs=training_time,
            n_training_samples=len(train_df),
            n_features=len(feature_columns),
            model_path=model_path,
            scaler_path=scaler_path,
            feature_importance=feature_importance,
            hyperparameters={
                'n_estimators': self.config.sklearn_n_estimators,
                'max_depth': self.config.sklearn_max_depth,
                'learning_rate': self.config.sklearn_learning_rate,
            },
        )
        
        logger.info(
            f"Sklearn training complete: {result.model_id} "
            f"AUC={result.auc:.4f} CV_AUC={result.cv_auc_mean:.4f}±{result.cv_auc_std:.4f}"
        )
        
        return result
    
    def predict(
        self,
        model_path: str,
        data: pd.DataFrame,
        feature_columns: List[str],
        scaler_path: str = None,
    ) -> np.ndarray:
        """
        Make predictions using trained model.
        
        Args:
            model_path: Path to saved model
            data: Data to predict on
            feature_columns: Feature columns
            scaler_path: Path to saved scaler
            
        Returns:
            Array of predicted probabilities
        """
        # Load model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Load scaler
        if scaler_path is None:
            scaler_path = model_path.replace('.pkl', '_scaler.pkl')
        
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
        else:
            scaler = None
        
        # Prepare data
        X = data[feature_columns].fillna(0)
        
        if scaler:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X.values
        
        # Predict
        probs = model.predict_proba(X_scaled)[:, 1]
        
        return probs
    
    def predict_with_loaded(
        self,
        data: pd.DataFrame,
        feature_columns: List[str] = None,
    ) -> np.ndarray:
        """
        Make predictions using already-loaded model.
        
        Args:
            data: Data to predict on
            feature_columns: Feature columns (uses stored if None)
            
        Returns:
            Array of predicted probabilities
        """
        if self._model is None:
            raise ValueError("No model loaded. Call train() or load() first.")
        
        feature_columns = feature_columns or self._feature_columns
        X = data[feature_columns].fillna(0)
        
        if self._scaler:
            X_scaled = self._scaler.transform(X)
        else:
            X_scaled = X.values
        
        probs = self._model.predict_proba(X_scaled)[:, 1]
        
        return probs
    
    def load(self, model_path: str, scaler_path: str = None) -> None:
        """Load a saved model and scaler"""
        with open(model_path, 'rb') as f:
            self._model = pickle.load(f)
        
        if scaler_path is None:
            scaler_path = model_path.replace('.pkl', '_scaler.pkl')
        
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self._scaler = pickle.load(f)
        
        logger.info(f"Loaded Sklearn model from {model_path}")
    
    def _create_base_models(self) -> Dict[str, Any]:
        """Create base models for ensemble"""
        models = {}
        
        # XGBoost
        try:
            from xgboost import XGBClassifier
            models['xgb'] = XGBClassifier(
                n_estimators=self.config.sklearn_n_estimators,
                max_depth=self.config.sklearn_max_depth,
                learning_rate=self.config.sklearn_learning_rate,
                subsample=self.config.sklearn_subsample,
                colsample_bytree=self.config.sklearn_colsample_bytree,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss',
                n_jobs=-1,
            )
        except ImportError:
            logger.warning("XGBoost not available")
        
        # LightGBM
        try:
            from lightgbm import LGBMClassifier
            models['lgb'] = LGBMClassifier(
                n_estimators=self.config.sklearn_n_estimators,
                max_depth=self.config.sklearn_max_depth,
                learning_rate=self.config.sklearn_learning_rate,
                subsample=self.config.sklearn_subsample,
                colsample_bytree=self.config.sklearn_colsample_bytree,
                random_state=42,
                verbose=-1,
                n_jobs=-1,
            )
        except ImportError:
            logger.warning("LightGBM not available")
        
        # CatBoost
        try:
            from catboost import CatBoostClassifier
            models['cat'] = CatBoostClassifier(
                iterations=self.config.sklearn_n_estimators,
                depth=self.config.sklearn_max_depth,
                learning_rate=self.config.sklearn_learning_rate,
                random_state=42,
                verbose=False,
                thread_count=-1,
            )
        except ImportError:
            logger.warning("CatBoost not available")
        
        # Random Forest
        models['rf'] = RandomForestClassifier(
            n_estimators=self.config.sklearn_n_estimators,
            max_depth=self.config.sklearn_max_depth + 4,  # RF benefits from deeper trees
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )
        
        logger.info(f"Created base models: {list(models.keys())}")
        return models
    
    def _create_stacking_ensemble(
        self,
        base_models: Dict[str, Any],
    ) -> StackingClassifier:
        """Create stacking ensemble with logistic regression meta-learner"""
        estimators = [(name, model) for name, model in base_models.items()]
        
        return StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=42,
            ),
            cv=5,
            stack_method='predict_proba',
            n_jobs=-1,
        )
    
    def _create_voting_ensemble(
        self,
        base_models: Dict[str, Any],
    ) -> VotingClassifier:
        """Create soft voting ensemble"""
        estimators = [(name, model) for name, model in base_models.items()]
        
        return VotingClassifier(
            estimators=estimators,
            voting='soft',
            n_jobs=-1,
        )
    
    def _get_feature_importance(
        self,
        feature_columns: List[str],
    ) -> Dict[str, float]:
        """Get feature importance from tree-based models"""
        importance = {}
        
        try:
            # Try to get from stacking estimators
            if hasattr(self._model, 'estimators_'):
                for name, estimator in self._model.estimators_:
                    if hasattr(estimator, 'feature_importances_'):
                        imp = estimator.feature_importances_
                        for i, col in enumerate(feature_columns):
                            if col not in importance:
                                importance[col] = 0.0
                            importance[col] += imp[i]
                
                # Average importance
                n_models = len(self._model.estimators_)
                importance = {k: v / n_models for k, v in importance.items()}
            
            # Sort by importance
            importance = dict(sorted(
                importance.items(),
                key=lambda x: x[1],
                reverse=True
            ))
            
        except Exception as e:
            logger.warning(f"Failed to get feature importance: {e}")
        
        return importance
    
    def _save_model(self, sport_code: str, bet_type: str) -> str:
        """Save model to disk"""
        save_dir = self.model_dir / sport_code / bet_type
        save_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = save_dir / "model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(self._model, f)
        
        return str(model_path)
    
    def _save_scaler(self, sport_code: str, bet_type: str) -> str:
        """Save scaler to disk"""
        save_dir = self.model_dir / sport_code / bet_type
        save_dir.mkdir(parents=True, exist_ok=True)
        
        scaler_path = save_dir / "scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(self._scaler, f)
        
        return str(scaler_path)


class SklearnTrainerMock:
    """Mock trainer for testing"""
    
    def __init__(self, config: MLConfig = None, model_dir: str = None):
        self.config = config or default_ml_config
        self.model_dir = Path(model_dir or "./models/sklearn_mock")
    
    def train(
        self,
        train_df: pd.DataFrame,
        target_column: str,
        feature_columns: List[str],
        sport_code: str,
        bet_type: str,
        **kwargs,
    ) -> SklearnModelResult:
        """Mock training"""
        logger.info(f"Mock Sklearn training for {sport_code} {bet_type}")
        
        return SklearnModelResult(
            model_id=f"mock_sk_{sport_code}_{bet_type}",
            sport_code=sport_code,
            bet_type=bet_type,
            ensemble_type="stacking",
            base_models=['xgb', 'lgb', 'cat', 'rf'],
            auc=0.61 + np.random.random() * 0.08,
            accuracy=0.58 + np.random.random() * 0.07,
            cv_auc_mean=0.60 + np.random.random() * 0.06,
            cv_auc_std=0.02 + np.random.random() * 0.02,
            training_time_secs=90.0,
            n_training_samples=len(train_df),
            n_features=len(feature_columns),
            model_path=str(self.model_dir / f"{sport_code}_{bet_type}_mock.pkl"),
        )
    
    def predict(
        self,
        model_path: str,
        data: pd.DataFrame,
        feature_columns: List[str],
        **kwargs,
    ) -> np.ndarray:
        """Mock prediction"""
        n_samples = len(data)
        return np.random.beta(2, 2, n_samples)


def get_sklearn_trainer(
    config: MLConfig = None,
    use_mock: bool = False,
) -> Union[SklearnEnsembleTrainer, SklearnTrainerMock]:
    """
    Factory function to get Sklearn trainer.
    
    Args:
        config: ML configuration
        use_mock: Use mock trainer for testing
        
    Returns:
        Sklearn trainer instance
    """
    if use_mock:
        return SklearnTrainerMock(config)
    return SklearnEnsembleTrainer(config)
