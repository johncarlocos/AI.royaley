"""
ROYALEY - Training Service
Phase 2: ML Training Orchestrator

This service orchestrates the entire ML training pipeline:
1. Load and prepare features from database
2. Run walk-forward validation
3. Train models using H2O/Sklearn/AutoGluon
4. Calibrate probabilities
5. Save to model registry
6. Update database

Usage:
    from app.services.ml.training_service import TrainingService
    
    service = TrainingService()
    result = await service.train_model("NFL", "spread", "h2o")
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
        
        # Initialize feature engineer
        self._feature_engineer = FeatureEngineer()
        
        # Initialize ELO manager
        self._elo_manager = MultiSportELOManager()
        
        logger.info(f"TrainingService initialized with model_dir={self.model_dir}")
    
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
    
    async def train_model(
        self,
        sport_code: str,
        bet_type: str,
        framework: str = "h2o",
        max_runtime_secs: int = None,
        min_samples: int = 500,
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
                    result.error_message = f"Insufficient training data: {len(train_df) if train_df is not None else 0} samples (min: {min_samples})"
                    if training_run:
                        await self._update_training_run(
                            session, training_run, TaskStatus.FAILED, result.error_message
                        )
                    return result
                
                result.training_samples = len(train_df)
                result.feature_count = len(feature_columns)
                
                logger.info(f"Loaded {len(train_df)} samples with {len(feature_columns)} features")
                
                # Walk-forward validation (optional)
                wfv_result = None
                if use_walk_forward and len(train_df) > 1000:
                    logger.info("Running walk-forward validation...")
                    wfv_result = await self._run_walk_forward_validation(
                        train_df, feature_columns, target_column, framework
                    )
                    result.wfv_accuracy = wfv_result.overall_metrics.get("accuracy", 0.0) if wfv_result else 0.0
                    result.wfv_roi = wfv_result.overall_metrics.get("roi", 0.0) if wfv_result else 0.0
                
                # Split train/validation
                train_data, valid_data = self._split_data(train_df, test_size=0.2)
                
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
                        metrics={
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
        training_run.completed_at = datetime.now(timezone.utc)
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
    ) -> Tuple[Optional[pd.DataFrame], List[str], str]:
        """
        Prepare training data from database.
        
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
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train/validation chronologically."""
        if "game_date" in df.columns:
            df = df.sort_values("game_date")
        
        split_idx = int(len(df) * (1 - test_size))
        train_df = df.iloc[:split_idx].copy()
        valid_df = df.iloc[split_idx:].copy()
        
        return train_df, valid_df
    
    async def _run_walk_forward_validation(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        target_column: str,
        framework: str,
    ) -> Optional[WalkForwardResult]:
        """Run walk-forward validation."""
        try:
            validator = WalkForwardValidator(
                training_window_days=settings.WFV_TRAINING_WINDOW_DAYS,
                test_window_days=settings.WFV_TEST_WINDOW_DAYS,
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
            
            result = await validator.validate(
                df=df,
                feature_columns=feature_columns,
                target_column=target_column,
                trainer=trainer,
                date_column="game_date",
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
        else:
            raise ValueError(f"Unknown framework: {framework}")
    
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
        
        if wfv_result:
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
        """Cleanup resources."""
        if self._h2o_trainer:
            self._h2o_trainer.cleanup()


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
