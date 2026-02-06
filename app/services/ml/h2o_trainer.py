"""
ROYALEY - H2O AutoML Trainer
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
        self._last_model = None
        self._last_feature_columns = None
        self._target_mem_size = None  # Set before _init_h2o for adaptive memory
        self._h2o_port = None  # Unique port per instance
        
    def _find_free_port(self) -> int:
        """Find a free port for H2O JVM — enables parallel training."""
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            return s.getsockname()[1]
    
    def _get_nthreads(self) -> int:
        """Calculate threads per H2O job for parallel training.
        
        Container has 48 CPUs / 256GB. Target: up to 10 parallel jobs.
        Each job gets ~4 threads to avoid over-subscription.
        """
        import os
        total_cores = os.cpu_count() or 8
        # Allow up to 10 parallel jobs, each gets a fair share of cores
        threads_per_job = max(2, total_cores // 10)
        return min(threads_per_job, 8)  # Cap at 8 threads per job
    
    def _init_h2o(self, max_mem_size: str = None) -> None:
        """Initialize H2O cluster with adaptive memory sizing and crash recovery.
        
        Each training process gets its own H2O JVM on a unique port,
        enabling safe parallel training across multiple sports/bet types.
        """
        if self._h2o_initialized:
            # Check if H2O is actually alive (not just flagged as initialized)
            if self._is_h2o_alive():
                # If already running but we need different memory, restart
                if max_mem_size and max_mem_size != self._target_mem_size:
                    logger.info(f"Restarting H2O with new memory: {max_mem_size}")
                    self._shutdown_h2o()
                else:
                    return
            else:
                # H2O JVM died but flag is stale — force restart
                logger.warning("H2O server is dead (stale flag). Restarting...")
                self._h2o_initialized = False
                self._last_model = None
        
        try:
            import h2o
            self._h2o = h2o
            
            # Use adaptive memory or configured default
            mem_size = max_mem_size or self.config.h2o_max_mem_size
            self._target_mem_size = mem_size
            
            # Find a unique free port for this process (enables parallel training)
            port = self._find_free_port()
            self._h2o_port = port
            nthreads = self._get_nthreads()
            
            logger.info(f"Starting H2O JVM on port {port} with {mem_size} memory, {nthreads} threads")
            
            h2o.init(
                port=port,
                max_mem_size=mem_size,
                nthreads=nthreads,
                min_mem_size="2g",
                bind_to_localhost=True,
            )
            
            self._h2o_initialized = True
            logger.info(f"H2O initialized on port {port} with {mem_size} memory, {nthreads} threads")
            
            # Brief pause to let JVM stabilize
            import time
            time.sleep(1)
            
        except ImportError:
            logger.error("H2O not installed. Install with: pip install h2o")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize H2O: {e}")
            raise
    
    def _is_h2o_alive(self) -> bool:
        """Check if H2O JVM is still running."""
        if not self._h2o:
            return False
        try:
            # Quick cluster status check — throws if JVM is dead
            self._h2o.cluster().is_running()
            return True
        except Exception:
            return False
    
    def _get_adaptive_mem_size(self, n_samples: int, n_features: int) -> str:
        """Calculate appropriate H2O memory based on dataset size.
        
        Container has 256GB. With up to 10 parallel jobs, each gets 8-20GB.
        """
        if n_samples < 500:
            return "8g"
        elif n_samples < 2000:
            return "12g"
        elif n_samples < 10000:
            return "16g"
        else:
            return "20g"  # Per-job max for parallel safety
    
    def _get_adaptive_min_rows(self, n_samples: int) -> int:
        """Calculate appropriate min_rows based on training set size."""
        if n_samples < 200:
            return max(5, int(n_samples * 0.05))
        elif n_samples < 500:
            return max(10, int(n_samples * 0.05))
        elif n_samples < 1000:
            return max(20, int(n_samples * 0.03))
        else:
            return 100  # H2O default
    
    def _shutdown_h2o(self) -> None:
        """Shutdown H2O cluster safely (handles already-dead JVM)."""
        if self._h2o:
            try:
                self._h2o.remove_all()
            except:
                pass
            try:
                self._h2o.cluster().shutdown()
            except:
                pass
        self._h2o_initialized = False
        self._last_model = None
        self._target_mem_size = None
        self._h2o_port = None
    
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
        fast_mode: bool = False,
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
            fast_mode: Use fast settings for walk-forward validation
            
        Returns:
            H2OModelResult with trained model info
        """
        # Calculate adaptive memory based on dataset size
        adaptive_mem = self._get_adaptive_mem_size(len(train_df), len(feature_columns))
        self._init_h2o(max_mem_size=adaptive_mem)
        
        # Calculate adaptive min_rows to prevent GBM failures on small data
        n_train = len(train_df)
        adaptive_min_rows = self._get_adaptive_min_rows(n_train)
        
        # Fast mode for walk-forward validation folds - prioritize speed
        if fast_mode:
            max_models = max_models or 5
            max_runtime_secs = max_runtime_secs or 30
            nfolds = 3
            stopping_tolerance = 0.02
            stopping_rounds = 2
            # Fast mode: Only fast algorithms, NO StackedEnsemble (causes crashes on small data)
            # CRITICAL: Exclude GBM and XGBoost when dataset too small for default min_rows=100
            # GBM needs min_rows*2 weighted rows per CV fold (~200+), so need ~600+ total
            if n_train < 600:
                include_algos = ["GLM", "DRF"]
                logger.info(f"Small dataset ({n_train} < 600): using GLM + DRF only (GBM min_rows=100 incompatible)")
            else:
                include_algos = ["GBM", "XGBoost", "GLM", "DRF"]
        else:
            # FULL training mode - optimize for best predictions
            max_models = max_models or 20  # More models for better selection
            max_runtime_secs = max_runtime_secs or 300  # 5 minutes for thorough search
            nfolds = 5  # More folds for robust validation
            stopping_tolerance = 0.001
            stopping_rounds = 5
            # Full mode: Exclude GBM for small data, include StackedEnsemble for large data
            if n_train < 1000:
                include_algos = ["GLM", "DRF"]
                logger.info(f"Small dataset ({n_train} < 1000): using GLM + DRF only for full mode")
            else:
                include_algos = ["GBM", "XGBoost", "GLM", "DRF", "StackedEnsemble"]
        
        # Adapt nfolds to dataset size to prevent CV splits with too few samples
        if n_train < 100:
            nfolds = 2
        elif n_train < 300:
            nfolds = min(nfolds, 3)
            
        seed = seed or self.config.h2o_seed
        
        logger.info(
            f"Starting H2O AutoML training for {sport_code} {bet_type} "
            f"with {n_train} samples, min_rows={adaptive_min_rows}, "
            f"nfolds={nfolds}, mem={adaptive_mem} (fast_mode={fast_mode})"
        )
        
        start_time = datetime.now(timezone.utc)
        
        # Track H2O frames for cleanup
        frames_to_cleanup = []
        
        try:
            from h2o.automl import H2OAutoML
            
            # Filter low-variance features BEFORE training
            feature_columns = self._filter_low_variance_features(
                train_df, feature_columns
            )
            
            if len(feature_columns) == 0:
                raise ValueError("No features remaining after variance filtering")
            
            # Convert to H2O frames
            train_h2o = self._h2o.H2OFrame(train_df)
            frames_to_cleanup.append(train_h2o)
            
            # Set target as categorical for classification
            train_h2o[target_column] = train_h2o[target_column].asfactor()
            
            # Validation frame if provided
            valid_h2o = None
            if validation_df is not None:
                valid_h2o = self._h2o.H2OFrame(validation_df)
                frames_to_cleanup.append(valid_h2o)
                valid_h2o[target_column] = valid_h2o[target_column].asfactor()
            
            # Configure AutoML with dynamic settings
            # Use unique project name per training run to avoid model conflicts across folds
            import uuid
            unique_project = f"{sport_code}_{bet_type}_{uuid.uuid4().hex[:8]}"
            
            aml = H2OAutoML(
                max_models=max_models,
                max_runtime_secs=max_runtime_secs,
                seed=seed,
                nfolds=nfolds,
                balance_classes=True,
                sort_metric="AUC",
                stopping_metric="AUC",
                stopping_tolerance=stopping_tolerance,
                stopping_rounds=stopping_rounds,
                include_algos=include_algos,
                project_name=unique_project,
                exploitation_ratio=0.1 if n_train < 500 else 0.0,  # More exploration for small datasets
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
            
            # Store model in memory for predict_with_loaded
            self._last_model = best_model
            self._last_feature_columns = feature_columns
            
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
            
            # Extract metrics safely - H2O returns deeply nested lists like [[0.5]]
            def extract_scalar(value, default=0.0):
                """Recursively extract scalar from any nested structure."""
                if value is None:
                    return default
                # Unwrap nested lists/tuples
                depth = 0
                while isinstance(value, (list, tuple)) and depth < 10:
                    if len(value) == 0:
                        return default
                    value = value[0]
                    depth += 1
                try:
                    result = float(value)
                    if result != result:  # NaN check
                        return default
                    return result
                except (TypeError, ValueError):
                    return default
            
            # Extract each metric with try/except
            try:
                auc_val = extract_scalar(val_perf.auc(), 0.0)
            except:
                auc_val = 0.0
            
            try:
                mpce_val = extract_scalar(val_perf.mean_per_class_error(), 0.5)
            except:
                mpce_val = 0.5
            
            try:
                logloss_val = extract_scalar(val_perf.logloss(), 0.0)
            except:
                logloss_val = 0.0
            
            accuracy_val = max(0.0, min(1.0, 1.0 - mpce_val))
            
            # Create result
            result = H2OModelResult(
                model_id=model_id,
                sport_code=sport_code,
                bet_type=bet_type,
                algorithm=best_model.algo,
                auc=auc_val,
                accuracy=accuracy_val,
                log_loss=logloss_val,
                mean_per_class_error=mpce_val,
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
            error_msg = str(e)
            logger.error(f"H2O training failed: {error_msg}")
            # If H2O JVM died, mark as not initialized so next call restarts it
            if "died unexpectedly" in error_msg or "RIP" in error_msg or "Connection refused" in error_msg:
                logger.warning("H2O JVM crashed — marking for restart on next call")
                self._h2o_initialized = False
                self._last_model = None
            raise
        finally:
            # CRITICAL: Clean up H2O frames to prevent memory leaks
            for frame in frames_to_cleanup:
                try:
                    self._h2o.remove(frame)
                except:
                    pass
    
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
    
    def predict_with_loaded(
        self,
        data: pd.DataFrame,
        feature_columns: List[str],
    ) -> np.ndarray:
        """
        Make predictions using the last trained model (still in memory).
        Used by walk-forward validation to avoid save/load overhead.
        
        Args:
            data: Data to predict on
            feature_columns: Feature columns
            
        Returns:
            Array of predicted probabilities
        """
        if not hasattr(self, '_last_model') or self._last_model is None:
            raise ValueError("No model in memory. Call train() first.")
        
        try:
            # Convert to H2O frame
            data_h2o = self._h2o.H2OFrame(data[feature_columns])
            
            # Predict using in-memory model
            predictions = self._last_model.predict(data_h2o)
            
            # Get probability of positive class
            probs = predictions['p1'].as_data_frame()['p1'].values
            
            return probs
            
        except Exception as e:
            logger.error(f"H2O predict_with_loaded failed: {e}")
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
    
    def _filter_low_variance_features(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        variance_threshold: float = 0.001,
    ) -> List[str]:
        """
        Remove features with near-zero variance that H2O would drop anyway.
        Pre-filtering prevents H2O 'bad and constant' warnings and improves stability.
        
        Args:
            df: Training dataframe
            feature_columns: List of feature column names
            variance_threshold: Minimum variance to keep a feature
            
        Returns:
            Filtered list of feature columns
        """
        try:
            # Calculate variance for each feature
            variances = df[feature_columns].var()
            
            # Also check for constant columns (all same value)
            nunique = df[feature_columns].nunique()
            
            selected = []
            dropped = []
            
            for col in feature_columns:
                col_var = variances.get(col, 0)
                col_unique = nunique.get(col, 0)
                
                # Drop if: zero/near-zero variance OR only 1 unique value
                if col_unique <= 1 or col_var < variance_threshold:
                    dropped.append(col)
                else:
                    selected.append(col)
            
            if dropped:
                logger.info(
                    f"Pre-filtered {len(dropped)} low-variance features: "
                    f"{dropped[:10]}{'...' if len(dropped) > 10 else ''}"
                )
            
            if not selected:
                logger.warning(
                    "All features have low variance! Keeping top features by variance."
                )
                # Fallback: keep top 10 features by variance
                top_features = variances.nlargest(min(10, len(feature_columns)))
                selected = top_features.index.tolist()
            
            return selected
            
        except Exception as e:
            logger.warning(f"Feature variance filtering failed: {e}. Using all features.")
            return feature_columns

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
        """Cleanup H2O cluster with proper resource deallocation."""
        if self._h2o_initialized and self._h2o:
            try:
                # Remove last model reference
                if self._last_model:
                    try:
                        self._h2o.remove(self._last_model.model_id)
                    except:
                        pass
                    self._last_model = None
                
                # Remove all remaining frames and models
                try:
                    self._h2o.remove_all()
                except:
                    pass
                
                # Shutdown cluster
                self._h2o.cluster().shutdown()
                self._h2o_initialized = False
                self._target_mem_size = None
                logger.info("H2O cluster shutdown complete")
            except Exception as e:
                logger.warning(f"H2O cleanup error: {e}")
                self._h2o_initialized = False


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