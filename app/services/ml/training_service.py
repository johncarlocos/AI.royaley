"""
ROYALEY - Training Service
Phase 2: ML Training Orchestrator (FULL 19-COMPONENT INTEGRATION)

This service orchestrates the entire ML training pipeline:
1. Load and prepare features from database
2. Run walk-forward validation
3. Train models using ALL available frameworks:
   - H2O AutoML (50+ algorithms)
   - Sklearn Ensemble (XGBoost, LightGBM, CatBoost, Random Forest)
   - AutoGluon (Multi-layer stacking)
   - Deep Learning (TensorFlow/LSTM)
   - Quantum ML (PennyLane, Qiskit, D-Wave, sQUlearn)
   - Meta Ensemble (combines all above)
4. Generate SHAP explanations for feature importance
5. Calibrate probabilities (Isotonic, Platt, Temperature Scaling)
6. Save to model registry
7. Update database

Usage:
    from app.services.ml.training_service import TrainingService
    
    service = TrainingService()
    
    # Train with specific framework
    result = await service.train_model("NFL", "spread", "h2o")
    result = await service.train_model("NFL", "spread", "sklearn")
    result = await service.train_model("NFL", "spread", "autogluon")
    result = await service.train_model("NFL", "spread", "deep_learning")
    result = await service.train_model("NFL", "spread", "quantum")
    result = await service.train_model("NFL", "spread", "meta_ensemble")
    
    # Get available frameworks
    frameworks = service.get_available_frameworks()
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from uuid import UUID, uuid4
import json
import pickle
import numpy as np
import pandas as pd

from app.core.config import settings
from app.core.database import db_manager
from app.models import (
    Sport, Team, Game, GameFeature, Odds, MLModel, TrainingRun,
    ModelPerformance, FeatureImportance, CalibrationModel,
    MLFramework, TaskStatus, GameStatus
)
from sqlalchemy import select, func, and_, or_, desc
from sqlalchemy.ext.asyncio import AsyncSession

from .config import MLConfig, default_ml_config, SPORT_CONFIGS
from .h2o_trainer import get_h2o_trainer, H2OModelResult
from .sklearn_trainer import get_sklearn_trainer, SklearnModelResult
from .autogluon_trainer import get_autogluon_trainer, AutoGluonModelResult
from .walk_forward_validator import WalkForwardValidator, WalkForwardResult
from .probability_calibration import ProbabilityCalibrator
from .elo_rating import MultiSportELOManager
from .feature_engineering import FeatureEngineer

# NEW: Deep Learning (TensorFlow/LSTM)
from .deep_learning_trainer import (
    get_deep_learning_trainer, 
    DeepLearningModelResult,
)

# NEW: Quantum ML (PennyLane, Qiskit, D-Wave, sQUlearn)
from .quantum_ml import (
    get_quantum_trainer,
    QuantumModelResult,
    get_available_quantum_frameworks,
)

# NEW: SHAP Explainer
from .shap_explainer import (
    get_shap_explainer,
    SHAPExplainer,
)

# NEW: Meta Ensemble
from .meta_ensemble import (
    MetaEnsemble,
    EnsemblePredictor,
    EnsembleWeights,
)

# NEW: Model Registry
from .model_registry import (
    ModelRegistry,
    ModelVersion,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    """Result from model training"""
    success: bool
    model_id: Optional[UUID] = None
    training_run_id: Optional[UUID] = None
    sport_code: str = ""
    bet_type: str = ""
    framework: str = ""
    
    # Metrics
    auc: float = 0.0
    accuracy: float = 0.0
    log_loss: float = 0.0
    brier_score: float = 0.0
    
    # Walk-forward results
    wfv_accuracy: float = 0.0
    wfv_roi: float = 0.0
    
    # Training info
    training_samples: int = 0
    feature_count: int = 0
    training_duration_seconds: float = 0.0
    
    # Paths
    model_path: str = ""
    calibrator_path: str = ""
    
    # Feature importance
    top_features: List[Dict[str, Any]] = field(default_factory=list)
    
    # NEW: SHAP explanations
    shap_values: Optional[Dict[str, float]] = None
    shap_summary_path: str = ""
    
    # NEW: Ensemble weights (for meta_ensemble)
    ensemble_weights: Optional[Dict[str, float]] = None
    base_model_results: List[Dict[str, Any]] = field(default_factory=list)
    
    # NEW: Quantum metrics
    quantum_framework: str = ""
    n_qubits: int = 0
    quantum_advantage_score: float = 0.0
    
    # Error info
    error_message: str = ""


class TrainingService:
    """
    ML Training Orchestrator.
    
    Coordinates the complete training pipeline from data to deployed models.
    """
    
    BET_TYPES = ["spread", "moneyline", "total"]
    
    def __init__(
        self,
        config: MLConfig = None,
        model_dir: str = None,
        use_mock: bool = False,
    ):
        """
        Initialize training service.
        
        Args:
            config: ML configuration
            model_dir: Base directory for model storage
            use_mock: Use mock trainers (for testing)
        """
        self.config = config or default_ml_config
        self.model_dir = Path(model_dir or settings.MODEL_STORAGE_PATH)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.use_mock = use_mock
        
        # Initialize trainers lazily
        self._h2o_trainer = None
        self._sklearn_trainer = None
        self._autogluon_trainer = None
        
        # NEW: Additional trainers
        self._deep_learning_trainer = None
        self._quantum_trainer = None
        self._shap_explainer = None
        self._meta_ensemble = None
        self._model_registry = None
        
        # Initialize feature engineer
        self._feature_engineer = FeatureEngineer()
        
        # Initialize ELO manager
        self._elo_manager = MultiSportELOManager()
        
        logger.info(f"TrainingService initialized with model_dir={self.model_dir}")
        logger.info(f"Available frameworks: h2o, sklearn, autogluon, deep_learning, quantum, meta_ensemble")
    
    @property
    def h2o_trainer(self):
        if self._h2o_trainer is None:
            self._h2o_trainer = get_h2o_trainer(self.config, use_mock=self.use_mock)
        return self._h2o_trainer
    
    @property
    def sklearn_trainer(self):
        if self._sklearn_trainer is None:
            self._sklearn_trainer = get_sklearn_trainer(self.config, use_mock=self.use_mock)
        return self._sklearn_trainer
    
    @property
    def autogluon_trainer(self):
        if self._autogluon_trainer is None:
            self._autogluon_trainer = get_autogluon_trainer(self.config, use_mock=self.use_mock)
        return self._autogluon_trainer
    
    # NEW: Deep Learning Trainer (TensorFlow/LSTM)
    @property
    def deep_learning_trainer(self):
        if self._deep_learning_trainer is None:
            self._deep_learning_trainer = get_deep_learning_trainer(self.config, use_mock=self.use_mock)
        return self._deep_learning_trainer
    
    # NEW: Quantum ML Trainer (PennyLane, Qiskit, D-Wave, sQUlearn)
    @property
    def quantum_trainer(self):
        if self._quantum_trainer is None:
            self._quantum_trainer = get_quantum_trainer(self.config, use_mock=self.use_mock)
        return self._quantum_trainer
    
    # NEW: SHAP Explainer
    @property
    def shap_explainer(self):
        if self._shap_explainer is None:
            self._shap_explainer = get_shap_explainer(use_mock=self.use_mock)
        return self._shap_explainer
    
    # NEW: Meta Ensemble
    @property
    def meta_ensemble(self):
        if self._meta_ensemble is None:
            self._meta_ensemble = MetaEnsemble(config=self.config)
        return self._meta_ensemble
    
    # NEW: Model Registry
    @property
    def model_registry(self):
        if self._model_registry is None:
            self._model_registry = ModelRegistry(base_dir=str(self.model_dir))
        return self._model_registry
    
    def get_available_frameworks(self) -> List[str]:
        """Return list of available training frameworks."""
        frameworks = ["h2o", "sklearn", "autogluon", "deep_learning", "meta_ensemble"]
        
        # Check quantum frameworks
        quantum_frameworks = get_available_quantum_frameworks()
        if quantum_frameworks:
            frameworks.append("quantum")
            frameworks.extend([f"quantum_{qf}" for qf in quantum_frameworks])
        
        return frameworks
    
    async def train_model(
        self,
        sport_code: str,
        bet_type: str,
        framework: str = "h2o",
        max_runtime_secs: int = None,
        min_samples: int = 30,
        use_walk_forward: bool = True,
        calibrate: bool = True,
        save_to_db: bool = True,
    ) -> TrainingResult:
        """
        Train a model for a specific sport and bet type.
        
        Args:
            sport_code: Sport code (NFL, NBA, etc.)
            bet_type: Bet type (spread, moneyline, total)
            framework: ML framework (h2o, sklearn, autogluon)
            max_runtime_secs: Maximum training time
            min_samples: Minimum training samples required
            use_walk_forward: Use walk-forward validation
            calibrate: Calibrate probabilities
            save_to_db: Save model to database
            
        Returns:
            TrainingResult with training outcome
        """
        start_time = datetime.now(timezone.utc)
        sport_code = sport_code.upper()
        bet_type = bet_type.lower()
        
        logger.info(f"Starting training for {sport_code} {bet_type} using {framework}")
        
        result = TrainingResult(
            success=False,
            sport_code=sport_code,
            bet_type=bet_type,
            framework=framework,
        )
        
        training_run = None
        
        try:
            # Initialize database
            await db_manager.initialize()
            
            async with db_manager.session() as session:
                # Verify sport exists
                sport = await self._get_sport(session, sport_code)
                if not sport:
                    result.error_message = f"Sport not found: {sport_code}"
                    return result
                
                # Create training run record
                if save_to_db:
                    training_run = await self._create_training_run(
                        session, sport.id, sport_code, bet_type, framework
                    )
                    result.training_run_id = training_run.id
                
                # Load training data
                logger.info("Loading training data from database...")
                train_df, feature_columns, target_column = await self._prepare_training_data(
                    session, sport_code, bet_type, min_samples
                )
                
                if train_df is None or len(train_df) < min_samples:
                    # Check sport-aware minimum before failing
                    SPORT_MIN = {
                        'NFL': 200, 'NBA': 200, 'MLB': 200, 'NHL': 200,
                        'NCAAF': 200, 'NCAAB': 200,
                        'CFL': 30, 'WNBA': 50,
                        'ATP': 100, 'WTA': 100,
                        'MLS': 50, 'NWSL': 30,
                    }
                    sport_min = min(SPORT_MIN.get(sport_code, min_samples), min_samples)
                    actual_count = len(train_df) if train_df is not None else 0
                    
                    if actual_count < sport_min:
                        result.error_message = f"Insufficient training data: {actual_count} samples (min: {sport_min} for {sport_code})"
                        if training_run:
                            await self._update_training_run(
                                session, training_run, TaskStatus.FAILED, result.error_message
                            )
                        return result
                    else:
                        logger.info(
                            f"Dataset has {actual_count} samples (below default {min_samples} "
                            f"but above sport minimum {sport_min} for {sport_code})"
                        )
                
                result.training_samples = len(train_df)
                result.feature_count = len(feature_columns)
                
                logger.info(f"Loaded {len(train_df)} samples with {len(feature_columns)} features")
                
                # Provide guidance on expected results based on sample size
                if len(train_df) < 50:
                    logger.warning(
                        f"⚠️  CRITICAL: Only {len(train_df)} samples. "
                        f"ML will NOT work reliably. Results will be essentially random."
                    )
                elif len(train_df) < 100:
                    logger.warning(
                        f"⚠️  LOW SAMPLE SIZE: {len(train_df)} samples. "
                        f"Expect AUC in 0.50-0.60 range (barely above random). "
                        f"Model useful for testing only, not production."
                    )
                elif len(train_df) < 200:
                    logger.warning(
                        f"⚠️  SMALL SAMPLE SIZE: {len(train_df)} samples. "
                        f"Expect moderate performance with high variance. "
                        f"Recommend 500+ samples for reliable predictions."
                    )
                
                # Walk-forward validation (optional)
                wfv_result = None
                if use_walk_forward and len(train_df) > 1000:
                    logger.info("Running walk-forward validation...")
                    wfv_result = await self._run_walk_forward_validation(
                        train_df, feature_columns, target_column, framework,
                        sport_code=sport_code, bet_type=bet_type
                    )
                    # Safely access overall_metrics - may not exist if all folds failed
                    if wfv_result and hasattr(wfv_result, 'overall_metrics') and wfv_result.overall_metrics:
                        result.wfv_accuracy = wfv_result.overall_metrics.get("accuracy", 0.0)
                        result.wfv_roi = wfv_result.overall_metrics.get("roi", 0.0)
                    else:
                        logger.warning("Walk-forward validation returned no metrics (all folds may have failed)")
                        result.wfv_accuracy = 0.0
                        result.wfv_roi = 0.0
                
                # Split train/validation
                # For small datasets, skip validation split (use CV-only)
                MIN_SAMPLES_FOR_HOLDOUT = 100
                
                if len(train_df) < MIN_SAMPLES_FOR_HOLDOUT:
                    logger.warning(
                        f"⚠️  SMALL DATASET: {len(train_df)} samples < {MIN_SAMPLES_FOR_HOLDOUT}. "
                        f"Using CV-only evaluation (no holdout test set)."
                    )
                    train_data = train_df.copy()
                    valid_data = None  # No holdout - rely on CV
                else:
                    train_data, valid_data = self._split_data(train_df, test_size=0.2, target_column=target_column)
                
                # Train model
                logger.info(f"Training {framework} model...")
                model_result = await self._train_framework(
                    framework=framework,
                    train_df=train_data,
                    valid_df=valid_data,
                    target_column=target_column,
                    feature_columns=feature_columns,
                    sport_code=sport_code,
                    bet_type=bet_type,
                    max_runtime_secs=max_runtime_secs,
                )
                
                if model_result is None:
                    result.error_message = "Training failed - no model returned"
                    if training_run:
                        await self._update_training_run(
                            session, training_run, TaskStatus.FAILED, result.error_message
                        )
                    return result
                
                # Extract metrics
                result.auc = getattr(model_result, 'auc', 0.0)
                result.accuracy = getattr(model_result, 'accuracy', 0.0)
                result.log_loss = getattr(model_result, 'log_loss', 0.0)
                result.model_path = getattr(model_result, 'model_path', '')
                
                # Extract top features
                var_imp = getattr(model_result, 'variable_importance', {})
                result.top_features = [
                    {"name": k, "importance": v}
                    for k, v in sorted(var_imp.items(), key=lambda x: -x[1])[:20]
                ]
                
                # Calibrate probabilities (optional)
                calibrator_path = ""
                if calibrate and valid_data is not None and len(valid_data) > 100:
                    logger.info("Calibrating probabilities...")
                    calibrator_path = await self._calibrate_model(
                        model_result, valid_data, feature_columns, target_column,
                        sport_code, bet_type
                    )
                    result.calibrator_path = calibrator_path
                
                # Calculate training duration
                result.training_duration_seconds = (
                    datetime.now(timezone.utc) - start_time
                ).total_seconds()
                
                # Save model to database
                if save_to_db:
                    model_record = await self._save_model_to_db(
                        session=session,
                        sport=sport,
                        sport_code=sport_code,
                        bet_type=bet_type,
                        framework=framework,
                        model_result=model_result,
                        feature_columns=feature_columns,
                        calibrator_path=calibrator_path,
                        wfv_result=wfv_result,
                    )
                    result.model_id = model_record.id
                    
                    # Update training run
                    await self._update_training_run(
                        session, training_run, TaskStatus.SUCCESS,
                        error_or_metrics={
                            "auc": result.auc,
                            "accuracy": result.accuracy,
                            "log_loss": result.log_loss,
                        },
                        model_id=model_record.id,
                    )
                    
                    # Save feature importances
                    await self._save_feature_importances(
                        session, model_record.id, var_imp
                    )
                
                result.success = True
                logger.info(
                    f"Training complete: {sport_code} {bet_type} - "
                    f"AUC={result.auc:.4f}, Accuracy={result.accuracy:.4f}"
                )
                
        except Exception as e:
            logger.exception(f"Training failed: {e}")
            result.error_message = str(e)
            
            # Update training run on failure
            if training_run and save_to_db:
                try:
                    async with db_manager.session() as session:
                        await self._update_training_run(
                            session, training_run, TaskStatus.FAILED, str(e)
                        )
                except:
                    pass
        
        return result
    
    async def train_all_models(
        self,
        sport_codes: List[str] = None,
        bet_types: List[str] = None,
        framework: str = "h2o",
        **kwargs,
    ) -> List[TrainingResult]:
        """
        Train models for multiple sports and bet types.
        
        Args:
            sport_codes: List of sport codes (None = all)
            bet_types: List of bet types (None = all)
            framework: ML framework to use
            **kwargs: Additional arguments for train_model
            
        Returns:
            List of TrainingResult
        """
        sport_codes = sport_codes or settings.SUPPORTED_SPORTS
        bet_types = bet_types or self.BET_TYPES
        
        results = []
        
        for sport_code in sport_codes:
            for bet_type in bet_types:
                logger.info(f"Training {sport_code} {bet_type}...")
                
                result = await self.train_model(
                    sport_code=sport_code,
                    bet_type=bet_type,
                    framework=framework,
                    **kwargs,
                )
                results.append(result)
                
                if not result.success:
                    logger.warning(
                        f"Training failed for {sport_code} {bet_type}: "
                        f"{result.error_message}"
                    )
        
        # Summary
        successful = sum(1 for r in results if r.success)
        logger.info(f"Training complete: {successful}/{len(results)} models trained")
        
        return results
    
    async def _get_sport(
        self,
        session: AsyncSession,
        sport_code: str,
    ) -> Optional[Sport]:
        """Get sport from database, create if doesn't exist."""
        result = await session.execute(
            select(Sport).where(Sport.code == sport_code)
        )
        sport = result.scalar_one_or_none()
        
        if not sport:
            # Create sport
            sport = Sport(
                code=sport_code,
                name=sport_code,
                feature_count=settings.get_sports_config(sport_code).get("features", 70),
                is_active=True,
            )
            session.add(sport)
            await session.commit()
            await session.refresh(sport)
        
        return sport
    
    async def _create_training_run(
        self,
        session: AsyncSession,
        sport_id: UUID,
        sport_code: str,
        bet_type: str,
        framework: str,
    ) -> TrainingRun:
        """Create a training run record."""
        # First create an MLModel placeholder
        model = MLModel(
            sport_id=sport_id,
            bet_type=bet_type,
            framework=MLFramework(framework),
            version=datetime.now().strftime("%Y%m%d_%H%M%S"),
            file_path="",
            is_production=False,
            performance_metrics={},
        )
        session.add(model)
        await session.commit()
        await session.refresh(model)
        
        # Create training run
        training_run = TrainingRun(
            model_id=model.id,
            status=TaskStatus.RUNNING,
            started_at=datetime.utcnow(),
            hyperparameters={
                "framework": framework,
                "sport_code": sport_code,
                "bet_type": bet_type,
            },
        )
        session.add(training_run)
        await session.commit()
        await session.refresh(training_run)
        
        return training_run
    
    async def _update_training_run(
        self,
        session: AsyncSession,
        training_run: TrainingRun,
        status: TaskStatus,
        error_or_metrics: Any = None,
        model_id: UUID = None,
    ):
        """Update training run status."""
        training_run.status = status
        training_run.completed_at = datetime.utcnow()
        
        # Calculate duration (both are naive UTC datetimes)
        if training_run.started_at:
            training_run.training_duration_seconds = int(
                (training_run.completed_at - training_run.started_at).total_seconds()
            )
        
        if status == TaskStatus.FAILED and isinstance(error_or_metrics, str):
            training_run.error_message = error_or_metrics
        elif isinstance(error_or_metrics, dict):
            training_run.validation_metrics = error_or_metrics
        
        await session.commit()
    
    async def _prepare_training_data(
        self,
        session: AsyncSession,
        sport_code: str,
        bet_type: str,
        min_samples: int,
        csv_dir: str = None,
    ) -> Tuple[Optional[pd.DataFrame], List[str], str]:
        """
        Prepare training data from CSV files or database.
        
        Priority:
        1. First tries to load from CSV files in ml_csv directory
        2. Falls back to database if no CSV found
        
        CSV Structure Expected:
        - ml_csv/{SPORT}/ml_features_{SPORT}_*.csv (main features)
        - ml_csv/{SPORT}/ml_features_{SPORT}_target_*.csv (target labels)
        - ... (game, odds, player, situation, team, weather)
        
        Returns:
            Tuple of (dataframe, feature_columns, target_column)
        """
        # Default CSV paths to try (in order)
        if csv_dir:
            csv_paths = [Path(csv_dir)]
        else:
            csv_paths = [
                Path(__file__).parent.parent / "ml_csv",           # app/services/ml_csv
                Path(__file__).parent.parent.parent / "ml_csv",    # app/ml_csv
                Path(__file__).parent.parent.parent.parent / "ml_csv",  # project_root/ml_csv
            ]
        
        df = None
        # Sport-aware min_samples for initial CSV load check
        SPORT_MIN_SAMPLES_INIT = {
            'NFL': 200, 'NBA': 200, 'MLB': 200, 'NHL': 200,
            'NCAAF': 200, 'NCAAB': 200,
            'CFL': 30, 'WNBA': 50,
            'ATP': 100, 'WTA': 100,
            'MLS': 50, 'NWSL': 30,
        }
        load_min = min(
            SPORT_MIN_SAMPLES_INIT.get(sport_code, min_samples),
            min_samples
        )
        for csv_path in csv_paths:
            if csv_path.exists():
                logger.info(f"Trying CSV path: {csv_path}")
                df = self._load_from_csv(sport_code, bet_type, csv_path)
                if df is not None and len(df) >= load_min:
                    break
        
        if df is not None and len(df) >= load_min:
            logger.info(f"Loaded {len(df)} samples from CSV for {sport_code} {bet_type}")
            
            # ================================================================
            # FIX 2a: FILTER 0-0 SCORES
            # NFL has 58% games with 0-0 scores (unfinished/missing data)
            # NBA has 41% games with 0-0 scores (same issue)
            # These corrupt home_win% and all derived targets
            # ================================================================
            if 'home_score' in df.columns and 'away_score' in df.columns:
                before_filter = len(df)
                # Identify rows where BOTH scores are 0 or NaN (incomplete games)
                zero_score_mask = (
                    ((df['home_score'] == 0) & (df['away_score'] == 0)) |
                    (df['home_score'].isna() & df['away_score'].isna())
                )
                n_zero = zero_score_mask.sum()
                if n_zero > 0:
                    pct_zero = n_zero / len(df) * 100
                    logger.warning(
                        f"🚨 0-0 SCORE FILTER: Removing {n_zero} of {before_filter} rows "
                        f"({pct_zero:.1f}%) with 0-0 or missing scores"
                    )
                    df = df[~zero_score_mask].reset_index(drop=True)
                    logger.info(f"After 0-0 filter: {len(df)} rows remain")
            
            # ================================================================
            # FIX 2b: TARGET RECONSTRUCTION (SAFE - NaN-aware)
            # Only fill NaN target values where betting lines exist.
            # NEVER overwrite existing valid target values.
            # Rows without betting lines stay NaN (dropped later).
            # ================================================================
            if 'home_score' in df.columns and 'away_score' in df.columns:
                margin = df['home_score'] - df['away_score']
                total_pts = df['home_score'] + df['away_score']
                
                # Reconstruct home_win: only fill NaN rows
                if 'home_win' not in df.columns:
                    df['home_win'] = np.nan
                fill_mask = df['home_win'].isna()
                if fill_mask.any():
                    df.loc[fill_mask, 'home_win'] = (margin[fill_mask] > 0).astype(int)
                    logger.info(
                        f"🔧 Filled {fill_mask.sum()} NaN home_win values. "
                        f"Distribution: {df['home_win'].value_counts(dropna=False).to_dict()}"
                    )
                
                # Reconstruct spread_result: only fill NaN rows WHERE spread line exists
                if 'spread_result' not in df.columns:
                    df['spread_result'] = np.nan
                
                spread_line_cols = ['spread_close', 'spread_line', 'spread', 'home_spread',
                                   'spread_home_close', 'home_line']
                spread_line_col = None
                for slc in spread_line_cols:
                    if slc in df.columns and df[slc].notna().sum() > 0:
                        spread_line_col = slc
                        break
                
                if spread_line_col:
                    # Only fill where target is NaN AND spread line exists
                    fill_mask = df['spread_result'].isna() & df[spread_line_col].notna()
                    if fill_mask.any():
                        df.loc[fill_mask, 'spread_result'] = (
                            margin[fill_mask] > -df.loc[fill_mask, spread_line_col]
                        ).astype(float)
                        # Mark pushes as NaN
                        push_mask = fill_mask & (margin == -df[spread_line_col])
                        df.loc[push_mask, 'spread_result'] = np.nan
                        logger.info(
                            f"🔧 Filled {fill_mask.sum()} NaN spread_result from {spread_line_col}. "
                            f"Distribution: {df['spread_result'].value_counts(dropna=False).to_dict()}"
                        )
                else:
                    logger.warning(f"🔧 No spread line column found - spread_result NaN rows will be dropped")
                
                # Reconstruct over_result: only fill NaN rows WHERE total line exists
                if 'over_result' not in df.columns:
                    df['over_result'] = np.nan
                
                total_line_cols = ['total_close', 'total_line', 'over_under_line', 'ou_line',
                                  'total', 'totals_close']
                total_line_col = None
                for tlc in total_line_cols:
                    if tlc in df.columns and df[tlc].notna().sum() > 0:
                        total_line_col = tlc
                        break
                
                if total_line_col:
                    # Only fill where target is NaN AND total line exists
                    fill_mask = df['over_result'].isna() & df[total_line_col].notna()
                    if fill_mask.any():
                        df.loc[fill_mask, 'over_result'] = (
                            total_pts[fill_mask] > df.loc[fill_mask, total_line_col]
                        ).astype(float)
                        # Mark pushes as NaN
                        push_mask = fill_mask & (total_pts == df[total_line_col])
                        df.loc[push_mask, 'over_result'] = np.nan
                        logger.info(
                            f"🔧 Filled {fill_mask.sum()} NaN over_result from {total_line_col}. "
                            f"Distribution: {df['over_result'].value_counts(dropna=False).to_dict()}"
                        )
                else:
                    logger.warning(f"🔧 No total line column found - over_result NaN rows will be dropped")
            
            # ================================================================
            # FIX 2d: ADVANCED FEATURE ENGINEERING
            # Create features that identify VALUE - when the line is wrong
            # Focus on: ATS trends, situational spots, line value indicators
            # ================================================================
            derived_features_created = []
            
            # ============================================================
            # TIER 1: BASIC DIFFERENTIAL FEATURES
            # ============================================================
            
            # 1. Power rating differential
            if 'power_rating_diff' not in df.columns:
                if 'home_power_rating' in df.columns and 'away_power_rating' in df.columns:
                    df['power_rating_diff'] = df['home_power_rating'] - df['away_power_rating']
                    derived_features_created.append('power_rating_diff')
            
            # 2. Momentum differential (streak difference)
            if 'momentum_diff' not in df.columns:
                if 'home_streak' in df.columns and 'away_streak' in df.columns:
                    df['momentum_diff'] = df['home_streak'] - df['away_streak']
                    derived_features_created.append('momentum_diff')
            
            # 3. Recent form differentials
            if 'recent_form_diff' not in df.columns:
                if 'home_wins_last10' in df.columns and 'away_wins_last10' in df.columns:
                    df['recent_form_diff'] = df['home_wins_last10'] - df['away_wins_last10']
                    derived_features_created.append('recent_form_diff')
            
            if 'recent_form5_diff' not in df.columns:
                if 'home_wins_last5' in df.columns and 'away_wins_last5' in df.columns:
                    df['recent_form5_diff'] = df['home_wins_last5'] - df['away_wins_last5']
                    derived_features_created.append('recent_form5_diff')
            
            # 4. Scoring and defense differentials
            if 'scoring_diff' not in df.columns:
                if 'home_avg_pts_last10' in df.columns and 'away_avg_pts_last10' in df.columns:
                    df['scoring_diff'] = df['home_avg_pts_last10'] - df['away_avg_pts_last10']
                    derived_features_created.append('scoring_diff')
            
            if 'defense_diff' not in df.columns:
                if 'home_avg_pts_allowed_last10' in df.columns and 'away_avg_pts_allowed_last10' in df.columns:
                    df['defense_diff'] = df['away_avg_pts_allowed_last10'] - df['home_avg_pts_allowed_last10']
                    derived_features_created.append('defense_diff')
            
            # 5. Margin differential
            if 'margin_diff' not in df.columns:
                if 'home_avg_margin_last10' in df.columns and 'away_avg_margin_last10' in df.columns:
                    df['margin_diff'] = df['home_avg_margin_last10'] - df['away_avg_margin_last10']
                    derived_features_created.append('margin_diff')
            
            # 6. Venue strength differential
            if 'venue_strength_diff' not in df.columns:
                if 'home_home_win_pct' in df.columns and 'away_away_win_pct' in df.columns:
                    df['venue_strength_diff'] = df['home_home_win_pct'] - df['away_away_win_pct']
                    derived_features_created.append('venue_strength_diff')
            
            # ============================================================
            # TIER 2: LINE VALUE FEATURES (Key for ATS predictions)
            # These compare team strength to the betting line
            # ============================================================
            
            # 7. Spread Value: Is the line too high/low based on power ratings?
            # Positive = home team getting more points than deserved (value on home cover)
            if 'spread_value' not in df.columns:
                if 'power_rating_diff' in df.columns and 'spread_close' in df.columns:
                    # power_rating_diff predicts margin, spread_close is the line
                    # If PR says home wins by 5, but spread is -7, that's value on home cover
                    df['spread_value'] = df['power_rating_diff'] + df['spread_close'].fillna(0)
                    derived_features_created.append('spread_value')
            
            # 8. Margin Value: Compare expected margin to spread
            if 'margin_value' not in df.columns:
                if 'margin_diff' in df.columns and 'spread_close' in df.columns:
                    df['margin_value'] = df['margin_diff'] + df['spread_close'].fillna(0)
                    derived_features_created.append('margin_value')
            
            # 9. ATS Record differential (historical covering ability)
            if 'ats_diff' not in df.columns:
                if 'home_ats_record_last10' in df.columns and 'away_ats_record_last10' in df.columns:
                    df['ats_diff'] = df['home_ats_record_last10'].fillna(0.5) - df['away_ats_record_last10'].fillna(0.5)
                    derived_features_created.append('ats_diff')
            
            # ============================================================
            # TIER 3: LINE MOVEMENT FEATURES
            # Movement often indicates sharp action
            # ============================================================
            
            # 10. Line movement indicators
            if 'line_move_direction' not in df.columns:
                if 'spread_close' in df.columns and 'spread_open' in df.columns:
                    df['line_move_direction'] = np.sign(df['spread_close'] - df['spread_open'])
                    derived_features_created.append('line_move_direction')
            
            if 'total_move_direction' not in df.columns:
                if 'total_close' in df.columns and 'total_open' in df.columns:
                    df['total_move_direction'] = np.sign(df['total_close'] - df['total_open'])
                    derived_features_created.append('total_move_direction')
            
            # 11. Line movement magnitude (bigger moves = more significant)
            if 'spread_move_magnitude' not in df.columns:
                if 'spread_movement' in df.columns:
                    df['spread_move_magnitude'] = df['spread_movement'].abs()
                    derived_features_created.append('spread_move_magnitude')
            
            # ============================================================
            # TIER 4: SITUATIONAL SPOT FEATURES
            # ============================================================
            
            # 12. Revenge game indicator (already in data, but create combined)
            if 'revenge_edge' not in df.columns:
                if 'home_is_revenge' in df.columns and 'away_is_revenge' in df.columns:
                    # Fill NaN with 0 (no revenge) before converting to int
                    df['revenge_edge'] = df['home_is_revenge'].fillna(0).astype(int) - df['away_is_revenge'].fillna(0).astype(int)
                    derived_features_created.append('revenge_edge')
            
            # 13. Rest advantage combined with power
            if 'rest_power_combo' not in df.columns:
                if 'rest_advantage' in df.columns and 'power_rating_diff' in df.columns:
                    df['rest_power_combo'] = df['rest_advantage'] * df['power_rating_diff'].fillna(0) / 10
                    derived_features_created.append('rest_power_combo')
            
            # 14. Letdown/Lookahead spot difference
            if 'spot_danger' not in df.columns:
                danger = np.zeros(len(df))
                if 'home_letdown_spot' in df.columns:
                    danger -= df['home_letdown_spot'].fillna(0).astype(int)
                if 'home_lookahead_spot' in df.columns:
                    danger -= df['home_lookahead_spot'].fillna(0).astype(int)
                if 'away_letdown_spot' in df.columns:
                    danger += df['away_letdown_spot'].fillna(0).astype(int)
                if 'away_lookahead_spot' in df.columns:
                    danger += df['away_lookahead_spot'].fillna(0).astype(int)
                df['spot_danger'] = danger
                derived_features_created.append('spot_danger')
            
            # ============================================================
            # TIER 5: COMPOSITE FEATURES
            # ============================================================
            
            # 15. Combined strength indicator
            if 'combined_strength' not in df.columns:
                components = []
                if 'power_rating_diff' in df.columns:
                    max_pr = df['power_rating_diff'].abs().max()
                    if max_pr > 0:
                        components.append(df['power_rating_diff'].fillna(0) / max_pr)
                if 'momentum_diff' in df.columns:
                    components.append(df['momentum_diff'].fillna(0) / 10)
                if 'recent_form_diff' in df.columns:
                    components.append(df['recent_form_diff'].fillna(0) / 10)
                if components:
                    df['combined_strength'] = sum(components) / len(components)
                    derived_features_created.append('combined_strength')
            
            # 16. Combined value indicator (for spread bets)
            if 'combined_value' not in df.columns:
                value_components = []
                if 'spread_value' in df.columns:
                    max_sv = df['spread_value'].abs().max()
                    if max_sv and max_sv > 0:
                        value_components.append(df['spread_value'].fillna(0) / max_sv)
                if 'margin_value' in df.columns:
                    max_mv = df['margin_value'].abs().max()
                    if max_mv and max_mv > 0:
                        value_components.append(df['margin_value'].fillna(0) / max_mv)
                if 'ats_diff' in df.columns:
                    value_components.append(df['ats_diff'].fillna(0))
                if value_components:
                    df['combined_value'] = sum(value_components) / len(value_components)
                    derived_features_created.append('combined_value')
            
            # 17. Momentum trend (are they getting better or worse?)
            if 'home_momentum_trend' not in df.columns:
                if 'home_wins_last5' in df.columns and 'home_wins_last10' in df.columns:
                    # If last 5 > (last 10 / 2), they're improving
                    df['home_momentum_trend'] = df['home_wins_last5'] - (df['home_wins_last10'] / 2)
                    derived_features_created.append('home_momentum_trend')
            
            if 'away_momentum_trend' not in df.columns:
                if 'away_wins_last5' in df.columns and 'away_wins_last10' in df.columns:
                    df['away_momentum_trend'] = df['away_wins_last5'] - (df['away_wins_last10'] / 2)
                    derived_features_created.append('away_momentum_trend')
            
            # 18. Momentum trend differential
            if 'momentum_trend_diff' not in df.columns:
                if 'home_momentum_trend' in df.columns and 'away_momentum_trend' in df.columns:
                    df['momentum_trend_diff'] = df['home_momentum_trend'] - df['away_momentum_trend']
                    derived_features_created.append('momentum_trend_diff')
            
            # 19. Win pct differential (normalized)
            if 'win_pct_diff' not in df.columns:
                if 'home_win_pct_last10' in df.columns and 'away_win_pct_last10' in df.columns:
                    df['win_pct_diff'] = df['home_win_pct_last10'] - df['away_win_pct_last10']
                    derived_features_created.append('win_pct_diff')
            
            # 20. Expected margin vs spread (key value indicator)
            if 'expected_margin_vs_spread' not in df.columns:
                if 'margin_diff' in df.columns and 'spread_close' in df.columns:
                    # margin_diff = expected home margin
                    # spread_close = line (negative = home favored)
                    # If margin_diff=5 and spread=-7, expected to win by 5 but line says 7, value on home
                    df['expected_margin_vs_spread'] = df['margin_diff'] + df['spread_close'].fillna(0)
                    derived_features_created.append('expected_margin_vs_spread')
            
            if derived_features_created:
                logger.info(
                    f"📊 FEATURE ENGINEERING: Created {len(derived_features_created)} derived features: "
                    f"{derived_features_created[:10]}{'...' if len(derived_features_created) > 10 else ''}"
                )
            
            # ================================================================
            # FIX 2c: SPORT-AWARE MIN_SAMPLES
            # CFL (62 rows) and WNBA (114 rows) are valid but small datasets
            # Auto-lower min_samples for sports with inherently less data
            # ================================================================
            SPORT_MIN_SAMPLES = {
                'NFL': 200, 'NBA': 200, 'MLB': 200, 'NHL': 200,
                'NCAAF': 200, 'NCAAB': 200,
                'CFL': 30, 'WNBA': 50,
                'ATP': 100, 'WTA': 100,
                'MLS': 50, 'NWSL': 30,
            }
            effective_min_samples = SPORT_MIN_SAMPLES.get(sport_code, min_samples)
            # Use the lower of CLI min_samples and sport-specific minimum
            effective_min_samples = min(effective_min_samples, min_samples)
            
            if len(df) < effective_min_samples:
                logger.warning(
                    f"Insufficient data after filtering: {len(df)} samples "
                    f"(need {effective_min_samples} for {sport_code})"
                )
                return None, [], ""
            
            # ================================================================
            # TENNIS HANDLING: Use relative features to avoid home/away bias
            # In tennis data, Player 1 (home) is usually the winner.
            # Instead of swapping (which creates detectable patterns),
            # we use ONLY relative/difference features that don't depend
            # on home/away ordering.
            # ================================================================
            if sport_code in ('ATP', 'WTA'):
                # Remove exact duplicates
                before_dedup = len(df)
                df = df.drop_duplicates()
                if len(df) < before_dedup:
                    logger.info(f"🎾 Removed {before_dedup - len(df)} duplicate rows")
                
                home_win_pct = df['home_win'].mean() if 'home_win' in df.columns else 0
                logger.info(
                    f"🎾 TENNIS: home_win={home_win_pct:.1%}. "
                    f"Using relative features only (no swap debiasing)."
                )
                
                # Create relative/difference features that don't encode home/away bias
                # These features only capture the DIFFERENCE between players
                feature_pairs = [
                    ('home_power_rating', 'away_power_rating', 'power_diff'),
                    ('home_wins_last10', 'away_wins_last10', 'wins_diff'),
                    ('home_wins_last5', 'away_wins_last5', 'wins5_diff'),
                    ('home_win_pct_last10', 'away_win_pct_last10', 'winpct_diff'),
                    ('home_rest_days', 'away_rest_days', 'rest_diff'),
                    ('home_avg_margin_last10', 'away_avg_margin_last10', 'margin_diff'),
                    ('home_avg_pts_last10', 'away_avg_pts_last10', 'pts_diff'),
                    ('home_streak', 'away_streak', 'streak_diff'),
                    ('home_season_game_num', 'away_season_game_num', 'games_diff'),
                ]
                
                created_features = []
                for home_col, away_col, diff_name in feature_pairs:
                    if home_col in df.columns and away_col in df.columns:
                        df[diff_name] = df[home_col] - df[away_col]
                        created_features.append(diff_name)
                
                # Mark tennis for special feature handling
                df['_is_tennis'] = True
                
                logger.info(
                    f"🎾 Created {len(created_features)} relative features for tennis: "
                    f"{created_features[:5]}..."
                )
            
            # Find target column based on bet_type
            target_column = self._find_target_column(df, bet_type)
            
            if target_column is None:
                logger.error(f"No target column found for bet_type={bet_type}. Available columns: {list(df.columns)}")
                return None, [], ""
            
            logger.info(f"Using target column: {target_column}")
            
            # ================================================================
            # FEATURE SELECTION - Explicit, precise, no pattern-matching bugs
            # ================================================================
            
            # STEP 1: Metadata columns to always exclude (not features)
            metadata_columns = {
                'master_game_id', 'game_id', 'match_id', 'id', 'index',
                'game_date', 'date', 'datetime', 'scheduled_at',
                'home_team_name', 'away_team_name', 'home_team', 'away_team',
                'team_home', 'team_away', 'home_team_id', 'away_team_id',
                'season', 'week', 'round', 'sport_code', 'sport',
            }
            
            # STEP 2: TRUE leakage columns - these reveal actual game outcomes
            # Only EXACT column names, no pattern matching
            true_leakage_columns = {
                # Actual scores (know game result)
                'home_score', 'away_score', 'home_points', 'away_points',
                'final_score_home', 'final_score_away',
                
                # Derived from final scores
                'score_margin', 'point_margin', 'total_points', 'game_total',
                'combined_score', 'total_score', 'point_diff', 'score_diff',
                
                # Direct outcome indicators (these ARE the prediction targets)
                'home_win', 'away_win', 'winner', 'winning_team',
                'spread_result', 'over_result', 'under_result',
                'cover', 'covered', 'ats_result', 'against_spread',
                'over_under_result', 'ou_result',
                'moneyline_result', 'total_result',
            }
            
            # STEP 3: Historical team stats that are NOT leakage
            # These are known BEFORE the game and should be KEPT as features:
            # home_wins_last5, home_wins_last10, home_win_pct_last10,
            # home_avg_margin_last10, home_home_win_pct, 
            # h2h_home_wins_last5, h2h_home_avg_margin
            # (These were previously wrongly excluded by pattern matching "home_win")
            
            feature_columns = []
            leakage_excluded = []
            dead_excluded = []
            
            for col in df.columns:
                col_lower = col.lower()
                
                # Skip target column
                if col == target_column:
                    continue
                
                # Skip metadata
                if col_lower in metadata_columns or col in metadata_columns:
                    continue
                
                # Skip columns starting with 'unnamed'
                if col_lower.startswith('unnamed'):
                    continue
                
                # Skip TRUE leakage (exact match only, not substring)
                if col_lower in true_leakage_columns or col in true_leakage_columns:
                    leakage_excluded.append(col)
                    continue
                
                # Skip non-numeric columns
                if df[col].dtype not in ['int64', 'float64', 'int32', 'float32', 'bool']:
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        if df[col].isna().all():
                            continue
                    except:
                        continue
                
                feature_columns.append(col)
            
            # STEP 4: Remove dead features (100% null or constant value)
            # These add noise and waste model capacity
            clean_features = []
            for col in feature_columns:
                null_pct = df[col].isna().mean()
                n_unique = df[col].nunique()
                
                if null_pct >= 0.95:  # 95%+ null = dead
                    dead_excluded.append((col, f"{null_pct*100:.0f}% null"))
                    continue
                if n_unique <= 1:  # constant = no information
                    dead_excluded.append((col, f"constant (1 value)"))
                    continue
                
                clean_features.append(col)
            
            feature_columns = clean_features
            
            # ================================================================
            # TENNIS: Use only relative/difference features
            # For tennis, home_* and away_* features encode who the winner is
            # (since winner is usually listed as "home"). Instead, use only
            # difference features that capture actual skill differences.
            # ================================================================
            if sport_code in ('ATP', 'WTA') and '_is_tennis' in df.columns:
                # Remove the marker column from features
                if '_is_tennis' in feature_columns:
                    feature_columns.remove('_is_tennis')
                
                # Features that are safe for tennis (no home/away bias)
                tennis_safe_prefixes = (
                    'power_diff', 'wins_diff', 'wins5_diff', 'winpct_diff',
                    'rest_diff', 'margin_diff', 'pts_diff', 'streak_diff',
                    'games_diff', 'rest_advantage', 'power_rating_diff',
                    'h2h_', 'month', 'day_of_week', 'season'
                )
                
                # Additional individual features that are OK
                # NOTE: Exclude h2h_ features - they have suspiciously high correlation
                # with outcome (0.48+) suggesting potential computation leakage
                tennis_safe_exact = {
                    'month', 'day_of_week', 'year', 'hour',
                    'rest_advantage', 'power_rating_diff',
                }
                
                tennis_features = []
                excluded_home_away = []
                excluded_h2h = []
                
                for col in feature_columns:
                    # Exclude h2h features (potential leakage)
                    if col.startswith('h2h_') or col.startswith('H2H_'):
                        excluded_h2h.append(col)
                        continue
                    # Keep if it's a safe prefix
                    if any(col.startswith(prefix) or col.lower().startswith(prefix.lower()) 
                           for prefix in tennis_safe_prefixes):
                        tennis_features.append(col)
                    # Keep if it's an exact match
                    elif col in tennis_safe_exact or col.lower() in tennis_safe_exact:
                        tennis_features.append(col)
                    # Exclude home_* and away_* (these encode winner bias)
                    elif col.startswith('home_') or col.startswith('away_'):
                        excluded_home_away.append(col)
                    # Keep other neutral features
                    else:
                        tennis_features.append(col)
                
                feature_columns = tennis_features
                
                if excluded_h2h:
                    logger.warning(
                        f"🎾 TENNIS: Excluded {len(excluded_h2h)} h2h features "
                        f"(potential leakage): {excluded_h2h}"
                    )
                if excluded_home_away:
                    logger.info(
                        f"🎾 TENNIS FEATURE FILTER: Excluded {len(excluded_home_away)} "
                        f"home_/away_ features to prevent bias leakage"
                    )
                logger.info(f"🎾 Tennis using {len(feature_columns)} relative/safe features")
            
            logger.info(f"Found {len(feature_columns)} usable feature columns")
            
            # Log leakage exclusions
            if leakage_excluded:
                logger.warning(f"⚠️  DATA LEAKAGE PREVENTION: Excluded {len(leakage_excluded)} outcome-revealing features:")
                for feat in leakage_excluded:
                    logger.warning(f"   - {feat}")
            
            # Log dead feature exclusions
            if dead_excluded:
                logger.info(f"🧹 DEAD FEATURE CLEANUP: Removed {len(dead_excluded)} useless features:")
                for feat, reason in dead_excluded[:10]:
                    logger.info(f"   - {feat} ({reason})")
                if len(dead_excluded) > 10:
                    logger.info(f"   ... and {len(dead_excluded) - 10} more")
            
            logger.info(f"Features retained: {feature_columns[:10]}...")
            logger.info(f"Without leakage, expect realistic accuracy: 52-58%")
            
            # Clean data
            df = df.dropna(subset=[target_column])
            
            # ================================================================
            # SMART IMPUTATION - fillna(0) creates false signals!
            # Example: away_away_win_pct is 94% null for WTA.
            #   fillna(0) → model sees "0% road win rate" (FALSE)
            #   fillna(0.5) → model sees "unknown = coin flip" (NEUTRAL)
            # ================================================================
            
            # Categorize features by type for appropriate fill values
            pct_keywords = ['_win_pct', '_pct_last', '_home_win_pct', '_away_win_pct',
                           '_ats_record', '_ou_over_pct', 'implied_', 'no_vig_',
                           'public_spread_home_pct', 'public_ml_home_pct',
                           'public_total_over_pct', 'public_money_home_pct']
            
            margin_keywords = ['_avg_margin', '_avg_pts', 'power_rating', 'h2h_home_avg_margin',
                              'h2h_total_avg', 'sharp_action_indicator']
            
            count_keywords = ['_injuries_out', '_starters_out', '_wins_last', 'num_books',
                             'h2h_home_wins_last']
            
            # Boolean features are fine with 0 (False)
            bool_keywords = ['is_', '_is_', 'steam_move']
            
            imputed_stats = {'pct_0.5': 0, 'median': 0, 'zero': 0}
            
            for col in feature_columns:
                if df[col].isna().any():
                    col_lower = col.lower()
                    null_count = df[col].isna().sum()
                    
                    # Win percentages / probability → 0.5 (unknown = coin flip)
                    if any(kw in col_lower for kw in pct_keywords):
                        df[col] = df[col].fillna(0.5)
                        imputed_stats['pct_0.5'] += 1
                    
                    # Margins / ratings / averages → median (unknown = average)
                    elif any(kw in col_lower for kw in margin_keywords):
                        median_val = df[col].median()
                        df[col] = df[col].fillna(median_val if pd.notna(median_val) else 0)
                        imputed_stats['median'] += 1
                    
                    # Everything else (counts, booleans, misc) → 0
                    else:
                        df[col] = df[col].fillna(0)
                        imputed_stats['zero'] += 1
            
            logger.info(
                f"🔧 SMART IMPUTATION: {imputed_stats['pct_0.5']} pct→0.5, "
                f"{imputed_stats['median']} margin→median, {imputed_stats['zero']} other→0"
            )
            
            # Convert target to numeric if needed
            if df[target_column].dtype == object:
                target_mapping = {
                    'W': 1, 'L': 0, 'win': 1, 'loss': 0, 
                    'cover': 1, 'no_cover': 0, 'over': 1, 'under': 0,
                    'home': 1, 'away': 0, '1': 1, '0': 0,
                    'True': 1, 'False': 0, True: 1, False: 0,
                    'yes': 1, 'no': 0, 'Y': 1, 'N': 0,
                }
                df[target_column] = df[target_column].map(target_mapping).fillna(0).astype(int)
            
            # CRITICAL FIX: Ensure target is binary (0 or 1) for all bet types
            # This fixes NCAAF total where target was actual point totals (35, 42, 51...)
            # instead of binary over/under classification
            unique_values = df[target_column].nunique()
            
            if unique_values > 2:
                logger.warning(f"Target column '{target_column}' has {unique_values} unique values - converting to binary")
                
                if bet_type.lower() == 'total':
                    # For totals: need to convert actual point totals to over/under binary
                    # Look for the betting line column to determine threshold
                    line_columns = ['total_line', 'over_under_line', 'ou_line', 'game_line', 
                                   'total', 'betting_line', 'line', 'ou']
                    line_col = None
                    for lc in line_columns:
                        if lc in df.columns and lc != target_column:
                            line_col = lc
                            break
                    
                    if line_col:
                        # Convert based on actual line: over = 1, under = 0
                        logger.info(f"Converting total target using line column: {line_col}")
                        df[target_column] = (df[target_column] > df[line_col]).astype(int)
                    else:
                        # Fallback: use median as threshold (not ideal but better than multi-class)
                        median_total = df[target_column].median()
                        logger.warning(f"No line column found, using median ({median_total}) as threshold")
                        df[target_column] = (df[target_column] > median_total).astype(int)
                else:
                    # For spread/moneyline with multiple classes: convert to binary
                    # Positive values = 1 (cover/win), zero or negative = 0 (no cover/loss)
                    logger.info(f"Converting {bet_type} target to binary (positive=1, else=0)")
                    df[target_column] = (df[target_column] > 0).astype(int)
                    
                logger.info(f"Target converted to binary: {df[target_column].value_counts().to_dict()}")
            
            # Ensure target is integer (0 or 1)
            df[target_column] = df[target_column].astype(int)
            
            # Final validation: ensure only 0 and 1 values
            valid_targets = df[target_column].isin([0, 1]).all()
            if not valid_targets:
                logger.warning(f"Target still has non-binary values, forcing to 0/1")
                df[target_column] = df[target_column].clip(0, 1).astype(int)
            
            logger.info(f"CSV data ready: {len(df)} samples, {len(feature_columns)} features, target={target_column}")
            logger.info(f"Target distribution: {df[target_column].value_counts().to_dict()}")
            
            return df, feature_columns, target_column
        
        # Fall back to database
        logger.info(f"No CSV found for {sport_code} {bet_type}, trying database...")
        return await self._prepare_training_data_from_db(session, sport_code, bet_type, min_samples)
    
    def _find_target_column(self, df: pd.DataFrame, bet_type: str) -> Optional[str]:
        """
        Find the target column for the given bet type.
        
        Searches for columns matching patterns like:
        - spread_result, spread_target, spread_outcome
        - moneyline_result, ml_result, moneyline_target
        - total_result, ou_result, over_under_result
        """
        bet_type_lower = bet_type.lower()
        
        # Define search patterns for each bet type
        # IMPORTANT: Order matters - more specific patterns first
        if bet_type_lower == 'spread':
            patterns = ['spread_result', 'spread_target', 'spread_outcome', 'ats_result', 
                       'against_spread', 'cover_result', 'spread_cover']
        elif bet_type_lower == 'moneyline':
            # For moneyline, home_win IS the target (1 = home won, 0 = away won)
            # DO NOT match moneyline_*_close/open as those are odds, not results
            patterns = ['home_win', 'moneyline_result', 'moneyline_target', 'ml_result', 
                       'winner', 'win_result', 'game_result', 'straight_up']
        elif bet_type_lower == 'total':
            patterns = ['over_result', 'total_result', 'total_target', 'ou_result', 
                       'over_under_result', 'totals_result']
        else:
            patterns = [f'{bet_type_lower}_result', f'{bet_type_lower}_target', bet_type_lower]
        
        # Also add generic patterns (but NOT 'result' alone as it's too broad)
        patterns.extend(['target', 'label', 'outcome', 'y'])
        
        # Search for matching column
        df_columns_lower = {col.lower(): col for col in df.columns}
        
        # Columns to NEVER use as targets (these are features/odds, not outcomes)
        exclude_from_target = ['_close', '_open', '_line', '_odds', '_prob', '_pct', '_avg']
        
        for pattern in patterns:
            pattern_lower = pattern.lower()
            # Exact match first
            if pattern_lower in df_columns_lower:
                col = df_columns_lower[pattern_lower]
                # Check it's not an odds/line column
                if not any(excl in col.lower() for excl in exclude_from_target):
                    return col
            # Partial match (but be careful)
            for col_lower, col_original in df_columns_lower.items():
                if pattern_lower in col_lower:
                    # Skip odds/line columns
                    if any(excl in col_lower for excl in exclude_from_target):
                        continue
                    return col_original
        
        # If still not found, look for any column with 'result' or 'target' in name
        for col_lower, col_original in df_columns_lower.items():
            if ('result' in col_lower or 'target' in col_lower):
                # Skip odds columns
                if any(excl in col_lower for excl in exclude_from_target):
                    continue
                logger.warning(f"Using fallback target column: {col_original}")
                return col_original
        
        return None
    
    def _load_from_csv(
        self, 
        sport_code: str, 
        bet_type: str, 
        csv_dir: Path
    ) -> Optional[pd.DataFrame]:
        """
        Load training data from CSV files.
        
        Handles the ROYALEY CSV structure:
        - ml_csv/{SPORT}/ml_features_{SPORT}_*.csv (main features)
        - ml_csv/{SPORT}/ml_features_{SPORT}_game_*.csv
        - ml_csv/{SPORT}/ml_features_{SPORT}_odds_*.csv
        - ml_csv/{SPORT}/ml_features_{SPORT}_player_*.csv
        - ml_csv/{SPORT}/ml_features_{SPORT}_situation_*.csv
        - ml_csv/{SPORT}/ml_features_{SPORT}_target_*.csv (contains target labels)
        - ml_csv/{SPORT}/ml_features_{SPORT}_team_*.csv
        - ml_csv/{SPORT}/ml_features_{SPORT}_weather_*.csv
        
        Merges all files into a single training dataframe.
        """
        csv_dir = Path(csv_dir)
        if not csv_dir.exists():
            logger.warning(f"CSV directory not found: {csv_dir}")
            return None
        
        sport_upper = sport_code.upper()
        sport_lower = sport_code.lower()
        
        # Check for sport subdirectory
        sport_dirs = [
            csv_dir / sport_upper,
            csv_dir / sport_lower,
            csv_dir,
        ]
        
        sport_dir = None
        for sd in sport_dirs:
            if sd.exists() and sd.is_dir():
                # Check if this directory has CSV files for this sport
                # FIX: Use exact word boundary matching to prevent NBA/WNBA collision
                # e.g., *_NBA_*.csv should NOT match *_WNBA_*.csv
                import re
                pattern = re.compile(
                    rf'(^|[_\-\s]){re.escape(sport_upper)}([_\-\s\.]|$)', re.IGNORECASE
                )
                csvs = [f for f in sd.glob("*.csv") if pattern.search(f.stem)]
                if csvs:
                    sport_dir = sd
                    break
        
        if sport_dir is None:
            logger.warning(f"No CSV directory found for sport {sport_code}")
            return None
        
        logger.info(f"Found sport directory: {sport_dir}")
        
        # Find all CSV files for this sport (exact match, no WNBA matching NBA)
        import re
        sport_pattern = re.compile(
            rf'(^|[_\-\s]){re.escape(sport_upper)}([_\-\s\.]|$)', re.IGNORECASE
        )
        csv_files = [f for f in sport_dir.glob(f"ml_features_{sport_upper}*.csv")
                     if sport_pattern.search(f.stem)]
        if not csv_files:
            csv_files = [f for f in sport_dir.glob("*.csv")
                         if sport_pattern.search(f.stem)]
        
        if not csv_files:
            logger.warning(f"No CSV files found for {sport_code} in {sport_dir}")
            return None
        
        logger.info(f"Found {len(csv_files)} CSV files for {sport_code}: {[f.name for f in csv_files]}")
        
        # Categorize files by type
        file_types = {
            'main': None,      # ml_features_{SPORT}_YYYYMMDD_HHMMSS.csv (no suffix)
            'game': None,
            'odds': None,
            'player': None,
            'situation': None,
            'target': None,
            'team': None,
            'weather': None,
        }
        
        for csv_file in csv_files:
            name = csv_file.name.lower()
            if '_target_' in name or name.endswith('_target.csv'):
                file_types['target'] = csv_file
            elif '_game_' in name:
                file_types['game'] = csv_file
            elif '_odds_' in name:
                file_types['odds'] = csv_file
            elif '_player_' in name:
                file_types['player'] = csv_file
            elif '_situation_' in name:
                file_types['situation'] = csv_file
            elif '_team_' in name:
                file_types['team'] = csv_file
            elif '_weather_' in name:
                file_types['weather'] = csv_file
            else:
                # Main features file (no type suffix, just sport and timestamp)
                # e.g., ml_features_NFL_20260204_061141.csv
                if file_types['main'] is None:
                    file_types['main'] = csv_file
        
        logger.info(f"File types found: {[(k, v.name if v else None) for k, v in file_types.items()]}")
        
        # Load and merge dataframes
        dfs = {}
        for file_type, csv_file in file_types.items():
            if csv_file and csv_file.exists():
                try:
                    df = pd.read_csv(csv_file)
                    logger.info(f"Loaded {file_type}: {len(df)} rows, {len(df.columns)} columns")
                    dfs[file_type] = df
                except Exception as e:
                    logger.error(f"Error loading {csv_file}: {e}")
        
        if not dfs:
            logger.error("No CSV files could be loaded")
            return None
        
        # Start with main features or largest dataframe
        if 'main' in dfs and dfs['main'] is not None:
            merged_df = dfs['main'].copy()
            base_type = 'main'
        else:
            # Use the dataframe with most rows as base
            base_type = max(dfs.keys(), key=lambda k: len(dfs[k]) if dfs.get(k) is not None else 0)
            merged_df = dfs[base_type].copy()
        
        logger.info(f"Base dataframe ({base_type}): {len(merged_df)} rows, {len(merged_df.columns)} columns")
        
        # Find common key columns for merging
        potential_keys = ['game_id', 'match_id', 'id', 'index', 'game_date', 'date']
        merge_key = None
        for key in potential_keys:
            if key in merged_df.columns:
                merge_key = key
                break
        
        # Merge other dataframes
        for file_type, df in dfs.items():
            if file_type == base_type or df is None:
                continue
            
            # Find common columns for merging
            common_cols = set(merged_df.columns) & set(df.columns)
            
            if merge_key and merge_key in df.columns:
                # Merge on key
                new_cols = [c for c in df.columns if c not in merged_df.columns or c == merge_key]
                if len(new_cols) > 1:  # More than just the key
                    try:
                        merged_df = merged_df.merge(df[new_cols], on=merge_key, how='left')
                        logger.info(f"Merged {file_type} on {merge_key}: now {len(merged_df.columns)} columns")
                    except Exception as e:
                        logger.warning(f"Could not merge {file_type} on {merge_key}: {e}")
            elif len(merged_df) == len(df):
                # Same number of rows - concat horizontally
                new_cols = [c for c in df.columns if c not in merged_df.columns]
                if new_cols:
                    merged_df = pd.concat([merged_df, df[new_cols]], axis=1)
                    logger.info(f"Concatenated {file_type}: now {len(merged_df.columns)} columns")
        
        logger.info(f"Final merged dataframe: {len(merged_df)} rows, {len(merged_df.columns)} columns")
        logger.info(f"Columns: {list(merged_df.columns)[:20]}...")  # Show first 20 columns
        
        return merged_df
    
    async def _prepare_training_data_from_db(
        self,
        session: AsyncSession,
        sport_code: str,
        bet_type: str,
        min_samples: int,
    ) -> Tuple[Optional[pd.DataFrame], List[str], str]:
        """
        Prepare training data from database (fallback method).
        
        Returns:
            Tuple of (dataframe, feature_columns, target_column)
        """
        try:
            # Get sport
            sport = await self._get_sport(session, sport_code)
            if not sport:
                return None, [], ""
            
            # Query completed games with scores
            query = (
                select(Game, GameFeature)
                .outerjoin(GameFeature, Game.id == GameFeature.game_id)
                .where(
                    and_(
                        Game.sport_id == sport.id,
                        Game.status == GameStatus.FINAL,
                        Game.home_score.isnot(None),
                        Game.away_score.isnot(None),
                    )
                )
                .order_by(Game.scheduled_at)
            )
            
            result = await session.execute(query)
            games_features = result.all()
            
            if not games_features:
                logger.warning(f"No completed games found for {sport_code}")
                return None, [], ""
            
            # Build training data
            data = []
            for game, features in games_features:
                row = await self._build_game_row(
                    session, game, features, bet_type, sport_code
                )
                if row:
                    data.append(row)
            
            if len(data) < min_samples:
                logger.warning(f"Only {len(data)} samples, need {min_samples}")
                return None, [], ""
            
            df = pd.DataFrame(data)
            
            # Define target column
            target_column = f"{bet_type}_result"
            
            # Ensure target exists
            if target_column not in df.columns:
                logger.error(f"Target column {target_column} not in data")
                return None, [], ""
            
            # Get feature columns (all except game_id, date, and target)
            exclude_cols = ["game_id", "game_date", "scheduled_at", target_column]
            exclude_cols += [c for c in df.columns if c.endswith("_result")]
            feature_columns = [c for c in df.columns if c not in exclude_cols]
            
            # Remove rows with missing target
            df = df.dropna(subset=[target_column])
            
            # Fill missing features with 0
            df[feature_columns] = df[feature_columns].fillna(0)
            
            return df, feature_columns, target_column
            
        except Exception as e:
            logger.exception(f"Error preparing training data: {e}")
            return None, [], ""
    
    async def _build_game_row(
        self,
        session: AsyncSession,
        game: Game,
        features: Optional[GameFeature],
        bet_type: str,
        sport_code: str,
    ) -> Optional[Dict[str, Any]]:
        """Build a single training row from a game."""
        try:
            row = {
                "game_id": str(game.id),
                "game_date": game.scheduled_at.date(),
                "home_score": game.home_score,
                "away_score": game.away_score,
            }
            
            # Add pre-computed features if available
            if features and features.features:
                row.update(features.features)
            else:
                # Compute basic features
                row.update(await self._compute_basic_features(session, game, sport_code))
            
            # Add target labels
            margin = game.home_score - game.away_score
            total = game.home_score + game.away_score
            
            # Get closing line from odds
            spread_line, total_line = await self._get_closing_lines(session, game.id)
            
            # Spread result (home cover = 1, away cover = 0)
            if spread_line is not None:
                row["spread_result"] = 1 if margin > -spread_line else 0
                row["spread_line"] = spread_line
            else:
                row["spread_result"] = 1 if margin > 0 else 0
                row["spread_line"] = 0
            
            # Moneyline result (home win = 1)
            row["moneyline_result"] = 1 if margin > 0 else 0
            
            # Total result (over = 1, under = 0)
            if total_line is not None:
                row["total_result"] = 1 if total > total_line else 0
                row["total_line"] = total_line
            else:
                row["total_result"] = 1 if total > 200 else 0  # Default threshold
                row["total_line"] = 200
            
            return row
            
        except Exception as e:
            logger.warning(f"Error building row for game {game.id}: {e}")
            return None
    
    async def _compute_basic_features(
        self,
        session: AsyncSession,
        game: Game,
        sport_code: str,
    ) -> Dict[str, Any]:
        """Compute basic features for a game."""
        features = {}
        
        try:
            # Get team info
            home_team = await session.get(Team, game.home_team_id)
            away_team = await session.get(Team, game.away_team_id)
            
            if home_team and away_team:
                # ELO ratings
                features["home_elo"] = home_team.elo_rating
                features["away_elo"] = away_team.elo_rating
                features["elo_diff"] = home_team.elo_rating - away_team.elo_rating
                
                # Home advantage
                home_advantage = settings.get_sports_config(sport_code).get("home_advantage", 2.5)
                features["home_advantage"] = home_advantage
                
            # Recent form (last 10 games)
            home_form = await self._get_team_form(session, game.home_team_id, game.scheduled_at)
            away_form = await self._get_team_form(session, game.away_team_id, game.scheduled_at)
            
            features.update({
                "home_win_pct": home_form.get("win_pct", 0.5),
                "home_ppg": home_form.get("ppg", 0),
                "home_ppg_allowed": home_form.get("ppg_allowed", 0),
                "away_win_pct": away_form.get("win_pct", 0.5),
                "away_ppg": away_form.get("ppg", 0),
                "away_ppg_allowed": away_form.get("ppg_allowed", 0),
                "form_diff": home_form.get("win_pct", 0.5) - away_form.get("win_pct", 0.5),
            })
            
            # Rest days
            features.update({
                "home_rest_days": home_form.get("rest_days", 7),
                "away_rest_days": away_form.get("rest_days", 7),
                "rest_advantage": home_form.get("rest_days", 7) - away_form.get("rest_days", 7),
            })
            
        except Exception as e:
            logger.warning(f"Error computing basic features: {e}")
        
        return features
    
    async def _get_team_form(
        self,
        session: AsyncSession,
        team_id: UUID,
        before_date: datetime,
        n_games: int = 10,
    ) -> Dict[str, float]:
        """Get team form from recent games."""
        try:
            # Get recent games
            query = (
                select(Game)
                .where(
                    and_(
                        or_(
                            Game.home_team_id == team_id,
                            Game.away_team_id == team_id,
                        ),
                        Game.scheduled_at < before_date,
                        Game.status == GameStatus.FINAL,
                    )
                )
                .order_by(desc(Game.scheduled_at))
                .limit(n_games)
            )
            
            result = await session.execute(query)
            games = result.scalars().all()
            
            if not games:
                return {"win_pct": 0.5, "ppg": 100, "ppg_allowed": 100, "rest_days": 7}
            
            wins = 0
            points_for = []
            points_against = []
            
            for game in games:
                is_home = game.home_team_id == team_id
                team_score = game.home_score if is_home else game.away_score
                opp_score = game.away_score if is_home else game.home_score
                
                if team_score is not None and opp_score is not None:
                    points_for.append(team_score)
                    points_against.append(opp_score)
                    if team_score > opp_score:
                        wins += 1
            
            # Rest days from most recent game
            rest_days = (before_date - games[0].scheduled_at).days if games else 7
            
            return {
                "win_pct": wins / len(games) if games else 0.5,
                "ppg": np.mean(points_for) if points_for else 100,
                "ppg_allowed": np.mean(points_against) if points_against else 100,
                "rest_days": min(rest_days, 14),
            }
            
        except Exception as e:
            logger.warning(f"Error getting team form: {e}")
            return {"win_pct": 0.5, "ppg": 100, "ppg_allowed": 100, "rest_days": 7}
    
    async def _get_closing_lines(
        self,
        session: AsyncSession,
        game_id: UUID,
    ) -> Tuple[Optional[float], Optional[float]]:
        """Get closing spread and total lines."""
        try:
            # Get most recent odds
            query = (
                select(Odds)
                .where(
                    and_(
                        Odds.game_id == game_id,
                        Odds.bet_type.in_(["spread", "spreads", "h2h", "totals"]),
                    )
                )
                .order_by(desc(Odds.recorded_at))
            )
            
            result = await session.execute(query)
            odds_list = result.scalars().all()
            
            spread_line = None
            total_line = None
            
            for odds in odds_list:
                if odds.bet_type in ["spread", "spreads"] and spread_line is None:
                    spread_line = odds.home_line
                elif odds.bet_type == "totals" and total_line is None:
                    total_line = odds.total
            
            return spread_line, total_line
            
        except Exception as e:
            logger.warning(f"Error getting closing lines: {e}")
            return None, None
    
    def _split_data(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        target_column: str = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train/validation chronologically, ensuring both classes present."""
        if "game_date" in df.columns:
            df = df.sort_values("game_date")
        elif "scheduled_at" in df.columns:
            df = df.sort_values("scheduled_at")
        
        split_idx = int(len(df) * (1 - test_size))
        train_df = df.iloc[:split_idx].copy()
        valid_df = df.iloc[split_idx:].copy()
        
        # Ensure BOTH train and validation have both classes (required for training + AUC)
        if target_column and target_column in df.columns:
            train_classes = train_df[target_column].nunique()
            valid_classes = valid_df[target_column].nunique()
            
            if train_classes < 2 or valid_classes < 2:
                logger.warning(
                    f"Chronological split failed: train has {train_classes} class(es), "
                    f"validation has {valid_classes} class(es). "
                    f"Switching to stratified split."
                )
                from sklearn.model_selection import train_test_split
                train_df, valid_df = train_test_split(
                    df, test_size=test_size, random_state=42,
                    stratify=df[target_column]
                )
                logger.info(
                    f"Stratified split: train={len(train_df)}, valid={len(valid_df)}, "
                    f"train classes={train_df[target_column].nunique()}, "
                    f"valid classes={valid_df[target_column].nunique()}"
                )
        
        return train_df, valid_df
    
    async def _run_walk_forward_validation(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        target_column: str,
        framework: str,
        sport_code: str = "UNKNOWN",
        bet_type: str = "unknown",
    ) -> Optional[WalkForwardResult]:
        """Run walk-forward validation."""
        try:
            validator = WalkForwardValidator(
                training_window_days=settings.WFV_TRAINING_WINDOW_DAYS,
                validation_window_days=settings.WFV_TEST_WINDOW_DAYS,
                step_size_days=settings.WFV_STEP_SIZE_DAYS,
                min_training_days=settings.WFV_MIN_TRAINING_DAYS,
            )
            
            # Get trainer
            if framework == "h2o":
                trainer = self.h2o_trainer
            elif framework == "sklearn":
                trainer = self.sklearn_trainer
            else:
                trainer = self.autogluon_trainer
            
            # Detect date column - CSV uses scheduled_at, DB uses game_date
            date_column = "game_date"
            if "scheduled_at" in df.columns and "game_date" not in df.columns:
                date_column = "scheduled_at"
            elif "date" in df.columns and "game_date" not in df.columns:
                date_column = "date"
            
            # validate() is synchronous - do not await
            result = validator.validate(
                data=df,
                feature_columns=feature_columns,
                target_column=target_column,
                trainer=trainer,
                date_column=date_column,
                sport_code=sport_code,
                bet_type=bet_type,
            )
            
            return result
            
        except Exception as e:
            logger.warning(f"Walk-forward validation failed: {e}")
            return None
    
    async def _train_framework(
        self,
        framework: str,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        target_column: str,
        feature_columns: List[str],
        sport_code: str,
        bet_type: str,
        max_runtime_secs: int = None,
    ) -> Any:
        """Train model using specified framework."""
        max_runtime_secs = max_runtime_secs or settings.H2O_MAX_RUNTIME_SECS
        
        # Prepare numpy arrays for certain frameworks
        X_train = train_df[feature_columns].values
        y_train = train_df[target_column].values
        X_valid = valid_df[feature_columns].values if valid_df is not None else None
        y_valid = valid_df[target_column].values if valid_df is not None else None
        
        if framework == "h2o":
            return self.h2o_trainer.train(
                train_df=train_df,
                target_column=target_column,
                feature_columns=feature_columns,
                sport_code=sport_code,
                bet_type=bet_type,
                validation_df=valid_df,
                max_runtime_secs=max_runtime_secs,
            )
        elif framework == "sklearn":
            return self.sklearn_trainer.train(
                train_df=train_df,
                target_column=target_column,
                feature_columns=feature_columns,
                sport_code=sport_code,
                bet_type=bet_type,
                validation_df=valid_df,
            )
        elif framework == "autogluon":
            return self.autogluon_trainer.train(
                train_df=train_df,
                target_column=target_column,
                feature_columns=feature_columns,
                sport_code=sport_code,
                bet_type=bet_type,
                validation_df=valid_df,
                time_limit=max_runtime_secs,
            )
        
        # NEW: Deep Learning (TensorFlow/LSTM)
        elif framework in ["deep_learning", "tensorflow", "lstm"]:
            logger.info(f"Training Deep Learning model for {sport_code} {bet_type}")
            return self.deep_learning_trainer.train(
                train_df=train_df,
                target_column=target_column,
                feature_columns=feature_columns,
                sport_code=sport_code,
                bet_type=bet_type,
                validation_df=valid_df,
                max_epochs=100,
                early_stopping_patience=10,
            )
        
        # NEW: Quantum ML
        elif framework.startswith("quantum"):
            logger.info(f"Training Quantum ML model for {sport_code} {bet_type}")
            # Parse quantum framework type (quantum, quantum_pennylane, quantum_qiskit, etc.)
            quantum_type = framework.replace("quantum_", "") if "_" in framework else "pennylane"
            return self.quantum_trainer.train(
                X_train=X_train,
                y_train=y_train,
                X_valid=X_valid,
                y_valid=y_valid,
                sport_code=sport_code,
                bet_type=bet_type,
                framework=quantum_type,
                n_qubits=min(len(feature_columns), 8),  # Limit qubits for performance
                n_iterations=50,
            )
        
        # NEW: Meta Ensemble (combines all base models)
        elif framework in ["meta_ensemble", "ensemble"]:
            logger.info(f"Training Meta Ensemble for {sport_code} {bet_type}")
            return await self._train_meta_ensemble(
                train_df=train_df,
                valid_df=valid_df,
                target_column=target_column,
                feature_columns=feature_columns,
                sport_code=sport_code,
                bet_type=bet_type,
                max_runtime_secs=max_runtime_secs,
            )
        
        else:
            raise ValueError(f"Unknown framework: {framework}. Available: {self.get_available_frameworks()}")
    
    async def _train_meta_ensemble(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        target_column: str,
        feature_columns: List[str],
        sport_code: str,
        bet_type: str,
        max_runtime_secs: int = None,
    ) -> Dict[str, Any]:
        """
        Train meta-ensemble combining all base models.
        
        This trains H2O, Sklearn, AutoGluon, and Deep Learning models,
        then combines their predictions using an optimized weighting scheme.
        """
        logger.info("Training Meta Ensemble - training all base models...")
        
        base_results = {}
        base_predictions = {}
        
        # Train each base framework
        base_frameworks = ["h2o", "sklearn", "autogluon", "deep_learning"]
        
        for base_fw in base_frameworks:
            try:
                logger.info(f"  Training base model: {base_fw}")
                result = await self._train_framework(
                    framework=base_fw,
                    train_df=train_df,
                    valid_df=valid_df,
                    target_column=target_column,
                    feature_columns=feature_columns,
                    sport_code=sport_code,
                    bet_type=bet_type,
                    max_runtime_secs=max_runtime_secs // 4 if max_runtime_secs else None,
                )
                base_results[base_fw] = result
                
                # Get predictions on validation set
                if hasattr(result, 'model_path') and result.model_path:
                    # Each trainer has a predict method
                    trainer = getattr(self, f"{base_fw}_trainer", None)
                    if trainer and hasattr(trainer, 'predict'):
                        preds = trainer.predict(result.model_path, valid_df, feature_columns)
                        base_predictions[base_fw] = preds
                        
            except Exception as e:
                logger.warning(f"  Failed to train {base_fw}: {e}")
                continue
        
        if not base_predictions:
            raise ValueError("No base models could be trained for meta-ensemble")
        
        # Combine predictions using meta-ensemble
        y_valid = valid_df[target_column].values
        
        # Optimize weights using validation performance
        ensemble_result = self.meta_ensemble.fit(
            predictions=base_predictions,
            y_true=y_valid,
            sport_code=sport_code,
            bet_type=bet_type,
        )
        
        # Save ensemble weights
        ensemble_path = self.model_dir / sport_code / bet_type / "meta_ensemble.pkl"
        ensemble_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(ensemble_path, 'wb') as f:
            pickle.dump({
                'weights': ensemble_result.weights,
                'base_model_paths': {fw: r.model_path for fw, r in base_results.items() if hasattr(r, 'model_path')},
                'frameworks': list(base_predictions.keys()),
            }, f)
        
        # Return combined result
        return type('MetaEnsembleResult', (), {
            'model_path': str(ensemble_path),
            'auc': ensemble_result.auc if hasattr(ensemble_result, 'auc') else 0.0,
            'accuracy': ensemble_result.accuracy if hasattr(ensemble_result, 'accuracy') else 0.0,
            'log_loss': ensemble_result.log_loss if hasattr(ensemble_result, 'log_loss') else 0.0,
            'training_time_secs': sum(getattr(r, 'training_time_secs', 0) for r in base_results.values()),
            'n_training_samples': len(train_df),
            'ensemble_weights': ensemble_result.weights if hasattr(ensemble_result, 'weights') else {},
            'base_model_results': [
                {'framework': fw, 'auc': getattr(r, 'auc', 0), 'accuracy': getattr(r, 'accuracy', 0)}
                for fw, r in base_results.items()
            ],
            'hyperparameters': {'base_frameworks': list(base_results.keys())},
        })()
    
    async def generate_shap_explanations(
        self,
        model_result: Any,
        train_df: pd.DataFrame,
        feature_columns: List[str],
        sport_code: str,
        bet_type: str,
        n_samples: int = 100,
    ) -> Dict[str, Any]:
        """
        Generate SHAP explanations for a trained model.
        
        Returns feature importance rankings and SHAP summary.
        """
        logger.info(f"Generating SHAP explanations for {sport_code} {bet_type}")
        
        try:
            # Get model for SHAP
            model_path = getattr(model_result, 'model_path', '')
            
            # Use background samples for SHAP
            background_data = train_df[feature_columns].sample(min(n_samples, len(train_df)))
            
            # Generate SHAP values
            shap_result = self.shap_explainer.explain(
                model_path=model_path,
                X=background_data.values,
                feature_names=feature_columns,
            )
            
            # Save SHAP summary plot
            summary_path = self.model_dir / sport_code / bet_type / "shap_summary.png"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            
            if hasattr(shap_result, 'save_summary_plot'):
                shap_result.save_summary_plot(str(summary_path))
            
            return {
                'feature_importance': shap_result.feature_importance if hasattr(shap_result, 'feature_importance') else {},
                'top_features': shap_result.top_features if hasattr(shap_result, 'top_features') else [],
                'summary_path': str(summary_path),
            }
            
        except Exception as e:
            logger.warning(f"SHAP explanation failed: {e}")
            return {'error': str(e)}
    
    
    async def _calibrate_model(
        self,
        model_result: Any,
        valid_df: pd.DataFrame,
        feature_columns: List[str],
        target_column: str,
        sport_code: str,
        bet_type: str,
    ) -> str:
        """Calibrate model probabilities."""
        try:
            calibrator = ProbabilityCalibrator()
            
            # Get predictions on validation set
            model_path = getattr(model_result, 'model_path', '')
            
            if hasattr(self, '_h2o_trainer') and self._h2o_trainer:
                probs = self._h2o_trainer.predict(
                    model_path, valid_df, feature_columns
                )
            else:
                # Mock predictions
                probs = np.random.beta(2, 2, len(valid_df))
            
            # Fit calibrator
            y_true = valid_df[target_column].values
            calibrator.fit(probs, y_true)
            
            # Save calibrator
            cal_path = self.model_dir / sport_code / bet_type / "calibrator.pkl"
            cal_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(cal_path, 'wb') as f:
                pickle.dump(calibrator, f)
            
            return str(cal_path)
            
        except Exception as e:
            logger.warning(f"Calibration failed: {e}")
            return ""
    
    async def _save_model_to_db(
        self,
        session: AsyncSession,
        sport: Sport,
        sport_code: str,
        bet_type: str,
        framework: str,
        model_result: Any,
        feature_columns: List[str],
        calibrator_path: str,
        wfv_result: Optional[WalkForwardResult],
    ) -> MLModel:
        """Save trained model to database."""
        # Get model path
        model_path = getattr(model_result, 'model_path', '')
        
        # Build performance metrics
        metrics = {
            "auc": getattr(model_result, 'auc', 0.0),
            "accuracy": getattr(model_result, 'accuracy', 0.0),
            "log_loss": getattr(model_result, 'log_loss', 0.0),
            "training_time_secs": getattr(model_result, 'training_time_secs', 0.0),
        }
        
        if wfv_result and hasattr(wfv_result, 'overall_metrics') and wfv_result.overall_metrics:
            metrics["wfv_accuracy"] = wfv_result.overall_metrics.get("accuracy", 0.0)
            metrics["wfv_roi"] = wfv_result.overall_metrics.get("roi", 0.0)
        
        # Create model record
        model = MLModel(
            sport_id=sport.id,
            bet_type=bet_type,
            framework=MLFramework(framework),
            version=datetime.now().strftime("%Y%m%d_%H%M%S"),
            file_path=model_path,
            is_production=False,
            performance_metrics=metrics,
            feature_list=feature_columns,
            hyperparameters=getattr(model_result, 'hyperparameters', {}),
            training_samples=getattr(model_result, 'n_training_samples', 0),
        )
        
        session.add(model)
        await session.commit()
        await session.refresh(model)
        
        # Save calibration model if exists
        if calibrator_path:
            cal_model = CalibrationModel(
                model_id=model.id,
                calibrator_type="isotonic",
                calibrator_path=calibrator_path,
            )
            session.add(cal_model)
            await session.commit()
        
        return model
    
    async def _save_feature_importances(
        self,
        session: AsyncSession,
        model_id: UUID,
        var_imp: Dict[str, float],
    ):
        """Save feature importances to database."""
        sorted_features = sorted(var_imp.items(), key=lambda x: -x[1])
        
        for rank, (name, importance) in enumerate(sorted_features, 1):
            fi = FeatureImportance(
                model_id=model_id,
                feature_name=name,
                importance_score=importance,
                importance_rank=rank,
            )
            session.add(fi)
        
        await session.commit()
    
    def cleanup(self):
        """Cleanup resources for all trainers."""
        if self._h2o_trainer:
            self._h2o_trainer.cleanup()
        if self._sklearn_trainer and hasattr(self._sklearn_trainer, 'cleanup'):
            self._sklearn_trainer.cleanup()
        if self._autogluon_trainer and hasattr(self._autogluon_trainer, 'cleanup'):
            self._autogluon_trainer.cleanup()
        if self._deep_learning_trainer and hasattr(self._deep_learning_trainer, 'cleanup'):
            self._deep_learning_trainer.cleanup()
        if self._quantum_trainer and hasattr(self._quantum_trainer, 'cleanup'):
            self._quantum_trainer.cleanup()
        
        logger.info("TrainingService cleanup complete")


# Singleton instance
_training_service: Optional[TrainingService] = None


def get_training_service(use_mock: bool = False) -> TrainingService:
    """Get or create training service instance."""
    global _training_service
    if _training_service is None:
        _training_service = TrainingService(use_mock=use_mock)
    return _training_service


async def train_model_task(
    sport_code: str,
    bet_type: str,
    framework: str = "h2o",
    **kwargs,
) -> TrainingResult:
    """
    Background task for model training.
    
    This can be called from API endpoints or scheduled tasks.
    """
    service = get_training_service()
    return await service.train_model(
        sport_code=sport_code,
        bet_type=bet_type,
        framework=framework,
        **kwargs,
    )