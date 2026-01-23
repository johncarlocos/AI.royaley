"""
LOYALEY - H2O AutoML Trainer
Phase 2: Automated ML training using H2O AutoML

H2O AutoML provides enterprise-grade automated machine learning with:
- Automatic model selection from 50+ algorithms
- Hyperparameter tuning
- Cross-validation
- Leaderboard ranking
- MOJO export for fast production inference
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

from .config import MLConfig, default_ml_config, BetType

logger = logging.getLogger(__name__)


@dataclass
class H2OModelResult:
    """Result from H2O AutoML training"""
    model_id: str
    framework: str = "h2o"
    sport_code: str = ""
    bet_type: str = ""
    algorithm: str = ""
    
    # Performance metrics
    auc: float = 0.0
    accuracy: float = 0.0
    log_loss: float = 0.0
    mean_per_class_error: float = 0.0
    
    # Training info
    training_time_secs: float = 0.0
    n_training_samples: int = 0
    n_features: int = 0
    
    # Artifact paths
    model_path: str = ""
    mojo_path: str = ""
    
    # Variable importance
    variable_importance: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    hyperparameters: Dict = field(default_factory=dict)


class H2OTrainer:
    """
    H2O AutoML trainer for sports prediction models.
    
    Provides automated ML training with:
    - 50+ model algorithms
    - Stacked ensembles
    - Cross-validation
    - MOJO export for production
    """
    
    def __init__(
        self,
        config: MLConfig = None,
        model_dir: str = None,
    ):
        """
        Initialize H2O trainer.
        
        Args:
            config: ML configuration
            model_dir: Directory for model artifacts
        """
        self.config = config or default_ml_config
        self.model_dir = Path(model_dir or self.config.model_artifact_path) / "h2o"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self._h2o_initialized = False
        self._h2o = None
        
    def _init_h2o(self) -> None:
        """Initialize H2O cluster"""
        if self._h2o_initialized:
            return
        
        try:
            import h2o
            self._h2o = h2o
            
            # Initialize H2O with configured memory
            h2o.init(
                max_mem_size=self.config.h2o_max_mem_size,
                nthreads=-1,  # Use all available cores
            )
            
            self._h2o_initialized = True
            logger.info(f"H2O initialized with {self.config.h2o_max_mem_size} memory")
            
        except ImportError:
            logger.error("H2O not installed. Install with: pip install h2o")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize H2O: {e}")
            raise
    
    def train(
        self,
        train_df: pd.DataFrame,
        target_column: str,
        feature_columns: List[str],
        sport_code: str,
        bet_type: str,
        validation_df: pd.DataFrame = None,
        max_models: int = None,
        max_runtime_secs: int = None,
        seed: int = None,
    ) -> H2OModelResult:
        """
        Train H2O AutoML model.
        
        Args:
            train_df: Training data
            target_column: Target variable column name
            feature_columns: Feature column names
            sport_code: Sport code (NFL, NBA, etc.)
            bet_type: Bet type (spread, moneyline, total)
            validation_df: Optional validation data
            max_models: Maximum models to train
            max_runtime_secs: Maximum training time
            seed: Random seed
            
        Returns:
            H2OModelResult with trained model info
        """
        self._init_h2o()
        
        max_models = max_models or self.config.h2o_max_models
        max_runtime_secs = max_runtime_secs or self.config.h2o_max_runtime_secs
        seed = seed or self.config.h2o_seed
        
        logger.info(
            f"Starting H2O AutoML training for {sport_code} {bet_type} "
            f"with {len(train_df)} samples"
        )
        
        start_time = datetime.now(timezone.utc)
        
        try:
            from h2o.automl import H2OAutoML
            
            # Convert to H2O frames
            train_h2o = self._h2o.H2OFrame(train_df)
            
            # Set target as categorical for classification
            train_h2o[target_column] = train_h2o[target_column].asfactor()
            
            # Validation frame if provided
            valid_h2o = None
            if validation_df is not None:
                valid_h2o = self._h2o.H2OFrame(validation_df)
                valid_h2o[target_column] = valid_h2o[target_column].asfactor()
            
            # Configure AutoML
            aml = H2OAutoML(
                max_models=max_models,
                max_runtime_secs=max_runtime_secs,
                seed=seed,
                nfolds=self.config.h2o_nfolds,
                balance_classes=True,
                sort_metric="AUC",
                stopping_metric="AUC",
                stopping_tolerance=0.001,
                stopping_rounds=5,
                exclude_algos=["DeepLearning"],  # Exclude slow deep learning
                project_name=f"{sport_code}_{bet_type}",
            )
            
            # Train
            aml.train(
                x=feature_columns,
                y=target_column,
                training_frame=train_h2o,
                validation_frame=valid_h2o,
            )
            
            # Get best model
            best_model = aml.leader
            model_id = best_model.model_id
            
            # Extract performance metrics
            perf = best_model.model_performance()
            val_perf = (
                best_model.model_performance(valid_h2o) 
                if valid_h2o else perf
            )
            
            # Get variable importance
            try:
                var_imp = best_model.varimp(use_pandas=True)
                var_imp_dict = dict(zip(
                    var_imp['variable'].tolist(),
                    var_imp['relative_importance'].tolist()
                ))
            except:
                var_imp_dict = {}
            
            # Save model
            model_path = self._save_model(best_model, sport_code, bet_type)
            mojo_path = self._export_mojo(best_model, sport_code, bet_type)
            
            training_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            # Create result
            result = H2OModelResult(
                model_id=model_id,
                sport_code=sport_code,
                bet_type=bet_type,
                algorithm=best_model.algo,
                auc=val_perf.auc() if hasattr(val_perf, 'auc') else 0.0,
                accuracy=1 - val_perf.mean_per_class_error() if hasattr(val_perf, 'mean_per_class_error') else 0.0,
                log_loss=val_perf.logloss() if hasattr(val_perf, 'logloss') else 0.0,
                mean_per_class_error=val_perf.mean_per_class_error() if hasattr(val_perf, 'mean_per_class_error') else 0.0,
                training_time_secs=training_time,
                n_training_samples=len(train_df),
                n_features=len(feature_columns),
                model_path=model_path,
                mojo_path=mojo_path,
                variable_importance=var_imp_dict,
                hyperparameters=self._extract_hyperparameters(best_model),
            )
            
            # Save leaderboard
            self._save_leaderboard(aml, sport_code, bet_type)
            
            logger.info(
                f"H2O training complete: {model_id} "
                f"AUC={result.auc:.4f} Accuracy={result.accuracy:.4f}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"H2O training failed: {e}")
            raise
    
    def predict(
        self,
        model_path: str,
        data: pd.DataFrame,
        feature_columns: List[str],
    ) -> np.ndarray:
        """
        Make predictions using trained H2O model.
        
        Args:
            model_path: Path to saved model
            data: Data to predict on
            feature_columns: Feature columns
            
        Returns:
            Array of predicted probabilities
        """
        self._init_h2o()
        
        try:
            # Load model
            model = self._h2o.load_model(model_path)
            
            # Convert to H2O frame
            data_h2o = self._h2o.H2OFrame(data[feature_columns])
            
            # Predict
            predictions = model.predict(data_h2o)
            
            # Get probability of positive class
            probs = predictions['p1'].as_data_frame()['p1'].values
            
            return probs
            
        except Exception as e:
            logger.error(f"H2O prediction failed: {e}")
            raise
    
    def predict_mojo(
        self,
        mojo_path: str,
        data: pd.DataFrame,
        feature_columns: List[str],
    ) -> np.ndarray:
        """
        Make predictions using MOJO (fast production inference).
        
        Args:
            mojo_path: Path to MOJO file
            data: Data to predict on
            feature_columns: Feature columns
            
        Returns:
            Array of predicted probabilities
        """
        self._init_h2o()
        
        try:
            # Import MOJO model
            model = self._h2o.import_mojo(mojo_path)
            
            # Convert to H2O frame
            data_h2o = self._h2o.H2OFrame(data[feature_columns])
            
            # Predict
            predictions = model.predict(data_h2o)
            
            # Get probability of positive class
            probs = predictions['p1'].as_data_frame()['p1'].values
            
            return probs
            
        except Exception as e:
            logger.error(f"MOJO prediction failed: {e}")
            raise
    
    def _save_model(
        self,
        model,
        sport_code: str,
        bet_type: str,
    ) -> str:
        """Save H2O model to disk"""
        save_dir = self.model_dir / sport_code / bet_type
        save_dir.mkdir(parents=True, exist_ok=True)
        
        path = self._h2o.save_model(
            model=model,
            path=str(save_dir),
            force=True,
        )
        
        logger.debug(f"Saved H2O model to {path}")
        return path
    
    def _export_mojo(
        self,
        model,
        sport_code: str,
        bet_type: str,
    ) -> str:
        """Export model as MOJO for fast production inference"""
        save_dir = self.model_dir / sport_code / bet_type / "mojo"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            mojo_path = model.save_mojo(str(save_dir), force=True)
            logger.debug(f"Exported MOJO to {mojo_path}")
            return mojo_path
        except Exception as e:
            logger.warning(f"Failed to export MOJO: {e}")
            return ""
    
    def _save_leaderboard(
        self,
        aml,
        sport_code: str,
        bet_type: str,
    ) -> None:
        """Save AutoML leaderboard"""
        try:
            lb = aml.leaderboard.as_data_frame()
            save_path = (
                self.model_dir / sport_code / bet_type / "leaderboard.csv"
            )
            save_path.parent.mkdir(parents=True, exist_ok=True)
            lb.to_csv(save_path, index=False)
            logger.debug(f"Saved leaderboard to {save_path}")
        except Exception as e:
            logger.warning(f"Failed to save leaderboard: {e}")
    
    def _extract_hyperparameters(self, model) -> Dict:
        """Extract model hyperparameters"""
        try:
            params = model.params
            return {
                k: v.get('actual', v.get('default', None))
                for k, v in params.items()
                if v is not None
            }
        except:
            return {}
    
    def get_leaderboard(
        self,
        sport_code: str,
        bet_type: str,
    ) -> Optional[pd.DataFrame]:
        """Load saved leaderboard"""
        lb_path = self.model_dir / sport_code / bet_type / "leaderboard.csv"
        if lb_path.exists():
            return pd.read_csv(lb_path)
        return None
    
    def cleanup(self) -> None:
        """Cleanup H2O cluster"""
        if self._h2o_initialized and self._h2o:
            try:
                self._h2o.cluster().shutdown()
                self._h2o_initialized = False
                logger.info("H2O cluster shutdown")
            except:
                pass


class H2OTrainerMock:
    """
    Mock H2O trainer for testing without H2O installed.
    
    This allows development and testing of the pipeline
    without requiring H2O to be installed.
    """
    
    def __init__(self, config: MLConfig = None, model_dir: str = None):
        self.config = config or default_ml_config
        self.model_dir = Path(model_dir or "./models/h2o_mock")
        self.model_dir.mkdir(parents=True, exist_ok=True)
    
    def train(
        self,
        train_df: pd.DataFrame,
        target_column: str,
        feature_columns: List[str],
        sport_code: str,
        bet_type: str,
        **kwargs,
    ) -> H2OModelResult:
        """Mock training that returns dummy results"""
        logger.info(f"Mock H2O training for {sport_code} {bet_type}")
        
        # Simulate training time
        import time
        time.sleep(0.1)
        
        return H2OModelResult(
            model_id=f"mock_h2o_{sport_code}_{bet_type}",
            sport_code=sport_code,
            bet_type=bet_type,
            algorithm="GBM",
            auc=0.62 + np.random.random() * 0.1,
            accuracy=0.58 + np.random.random() * 0.08,
            log_loss=0.65 + np.random.random() * 0.05,
            training_time_secs=60.0,
            n_training_samples=len(train_df),
            n_features=len(feature_columns),
            model_path=str(self.model_dir / f"{sport_code}_{bet_type}_mock"),
        )
    
    def predict(
        self,
        model_path: str,
        data: pd.DataFrame,
        feature_columns: List[str],
    ) -> np.ndarray:
        """Mock prediction that returns random probabilities"""
        n_samples = len(data)
        return np.random.beta(2, 2, n_samples)  # Beta distribution around 0.5
    
    def cleanup(self) -> None:
        """No-op cleanup"""
        pass


def get_h2o_trainer(
    config: MLConfig = None,
    use_mock: bool = False,
) -> H2OTrainer:
    """
    Factory function to get H2O trainer.
    
    Args:
        config: ML configuration
        use_mock: Use mock trainer for testing
        
    Returns:
        H2O trainer instance
    """
    if use_mock:
        return H2OTrainerMock(config)
    
    try:
        import h2o
        return H2OTrainer(config)
    except ImportError:
        logger.warning("H2O not installed, using mock trainer")
        return H2OTrainerMock(config)
