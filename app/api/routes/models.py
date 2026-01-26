"""
ROYALEY - ML Models API Routes
Model management, training, and performance tracking
"""

from datetime import datetime, timedelta
from typing import Optional, List
from enum import Enum

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from pydantic import BaseModel, Field
from sqlalchemy import select, func, and_, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_db
from app.api.dependencies import get_current_user, require_roles
from app.core.cache import CacheManager
from app.core.config import settings
from app.models import User, MLModel, ModelPerformance, TrainingRun, Sport

router = APIRouter()
cache = CacheManager()


# ============== Enums ==============

class ModelStatus(str, Enum):
    TRAINING = "training"
    READY = "ready"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    FAILED = "failed"


class ModelFramework(str, Enum):
    H2O = "h2o"
    AUTOGLUON = "autogluon"
    SKLEARN = "sklearn"
    META_ENSEMBLE = "meta_ensemble"


# ============== Schemas ==============

class ModelBase(BaseModel):
    """Base model schema"""
    id: int
    sport_code: str
    bet_type: str
    framework: str
    version: str
    status: str
    accuracy: Optional[float]
    auc: Optional[float]
    log_loss: Optional[float]
    created_at: datetime
    promoted_at: Optional[datetime]
    
    class Config:
        from_attributes = True


class ModelDetail(ModelBase):
    """Detailed model information"""
    feature_count: int
    training_samples: int
    validation_samples: int
    hyperparameters: Optional[dict]
    feature_importance: Optional[dict]
    calibration_method: Optional[str]
    training_duration_seconds: Optional[int]
    model_path: Optional[str]


class ModelPerformanceMetrics(BaseModel):
    """Model performance over time"""
    date: datetime
    predictions_count: int
    accuracy: float
    tier_a_accuracy: Optional[float]
    tier_b_accuracy: Optional[float]
    avg_clv: Optional[float]
    roi: Optional[float]


class TrainingRequest(BaseModel):
    """Model training request"""
    sport_code: str = Field(..., description="Sport code (e.g., NBA, NFL)")
    bet_type: str = Field(..., description="Bet type (spread, moneyline, total)")
    framework: ModelFramework = Field(default=ModelFramework.META_ENSEMBLE)
    max_runtime_seconds: int = Field(default=3600, ge=300, le=14400)
    use_gpu: bool = Field(default=True)
    hyperparameters: Optional[dict] = None


class TrainingRunResponse(BaseModel):
    """Training run response"""
    id: int
    sport_code: str
    bet_type: str
    framework: str
    status: str
    started_at: datetime
    completed_at: Optional[datetime]
    duration_seconds: Optional[int]
    best_model_id: Optional[int]
    metrics: Optional[dict]
    
    class Config:
        from_attributes = True


class ModelComparisonResponse(BaseModel):
    """Model comparison response"""
    models: List[ModelBase]
    comparison_metrics: dict
    recommendation: str


class FeatureImportanceResponse(BaseModel):
    """Feature importance response"""
    model_id: int
    sport_code: str
    bet_type: str
    features: List[dict]
    shap_available: bool


# ============== Endpoints ==============

@router.get("/", response_model=List[ModelBase])
async def list_models(
    sport_code: Optional[str] = None,
    bet_type: Optional[str] = None,
    framework: Optional[ModelFramework] = None,
    status: Optional[ModelStatus] = None,
    production_only: bool = False,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    List all ML models with optional filtering.
    
    - **sport_code**: Filter by sport
    - **bet_type**: Filter by bet type (spread, moneyline, total)
    - **framework**: Filter by ML framework
    - **status**: Filter by model status
    - **production_only**: Show only production models
    """
    query = select(MLModel)
    
    if sport_code:
        query = query.where(MLModel.sport_code == sport_code.upper())
    if bet_type:
        query = query.where(MLModel.bet_type == bet_type.lower())
    if framework:
        query = query.where(MLModel.framework == framework.value)
    if status:
        query = query.where(MLModel.status == status.value)
    if production_only:
        query = query.where(MLModel.status == "production")
    
    query = query.order_by(desc(MLModel.created_at))
    
    result = await db.execute(query)
    return result.scalars().all()


@router.get("/production", response_model=List[ModelBase])
async def list_production_models(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    List all models currently in production.
    
    Returns one model per sport/bet_type combination.
    """
    cache_key = "models:production"
    cached = await cache.get(cache_key)
    if cached:
        return cached
    
    result = await db.execute(
        select(MLModel).where(MLModel.status == "production")
        .order_by(MLModel.sport_code, MLModel.bet_type)
    )
    models = result.scalars().all()
    
    await cache.set(cache_key, [ModelBase.model_validate(m).model_dump() for m in models], ttl=300)
    
    return models


@router.get("/{model_id}", response_model=ModelDetail)
async def get_model_detail(
    model_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get detailed information about a specific model.
    
    Includes hyperparameters, feature importance, and training details.
    """
    result = await db.execute(
        select(MLModel).where(MLModel.id == model_id)
    )
    model = result.scalar_one_or_none()
    
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found"
        )
    
    return model


@router.get("/{model_id}/performance", response_model=List[ModelPerformanceMetrics])
async def get_model_performance(
    model_id: int,
    days: int = 30,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get model performance metrics over time.
    
    - **days**: Number of days of history (default: 30)
    """
    result = await db.execute(
        select(MLModel).where(MLModel.id == model_id)
    )
    if not result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found"
        )
    
    start_date = datetime.utcnow() - timedelta(days=days)
    
    result = await db.execute(
        select(ModelPerformance)
        .where(
            ModelPerformance.model_id == model_id,
            ModelPerformance.date >= start_date
        )
        .order_by(ModelPerformance.date)
    )
    
    return result.scalars().all()


@router.get("/{model_id}/features", response_model=FeatureImportanceResponse)
async def get_feature_importance(
    model_id: int,
    top_n: int = 20,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get feature importance for a model.
    
    - **top_n**: Number of top features to return (default: 20)
    """
    result = await db.execute(
        select(MLModel).where(MLModel.id == model_id)
    )
    model = result.scalar_one_or_none()
    
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found"
        )
    
    feature_importance = model.feature_importance or {}
    
    # Sort by importance and take top N
    sorted_features = sorted(
        feature_importance.items(),
        key=lambda x: abs(x[1]) if isinstance(x[1], (int, float)) else 0,
        reverse=True
    )[:top_n]
    
    features = [
        {
            "name": name,
            "importance": value,
            "rank": idx + 1
        }
        for idx, (name, value) in enumerate(sorted_features)
    ]
    
    return FeatureImportanceResponse(
        model_id=model_id,
        sport_code=model.sport_code,
        bet_type=model.bet_type,
        features=features,
        shap_available=model.framework in ["h2o", "autogluon", "sklearn"]
    )


@router.post("/train", response_model=TrainingRunResponse, status_code=status.HTTP_202_ACCEPTED)
async def train_model(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles(["admin", "system"]))
):
    """
    Trigger model training for a sport/bet_type combination.
    
    **Admin only**
    
    Training runs asynchronously. Use the training run ID to check status.
    """
    # Verify sport exists
    result = await db.execute(
        select(Sport).where(Sport.code == request.sport_code.upper())
    )
    if not result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid sport code: {request.sport_code}"
        )
    
    # Check for existing training run
    result = await db.execute(
        select(TrainingRun).where(
            TrainingRun.sport_code == request.sport_code.upper(),
            TrainingRun.bet_type == request.bet_type.lower(),
            TrainingRun.status == "running"
        )
    )
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Training already in progress for this sport/bet_type"
        )
    
    # Create training run record
    training_run = TrainingRun(
        sport_code=request.sport_code.upper(),
        bet_type=request.bet_type.lower(),
        framework=request.framework.value,
        status="pending",
        started_at=datetime.utcnow(),
        config={
            "max_runtime_seconds": request.max_runtime_seconds,
            "use_gpu": request.use_gpu,
            "hyperparameters": request.hyperparameters
        },
        initiated_by=current_user.id
    )
    
    db.add(training_run)
    await db.commit()
    await db.refresh(training_run)
    
    # Queue training task
    background_tasks.add_task(
        execute_training,
        training_run.id,
        request.sport_code.upper(),
        request.bet_type.lower(),
        request.framework.value,
        request.max_runtime_seconds,
        request.use_gpu,
        request.hyperparameters
    )
    
    return training_run


async def execute_training(
    training_run_id: int,
    sport_code: str,
    bet_type: str,
    framework: str,
    max_runtime: int,
    use_gpu: bool,
    hyperparameters: Optional[dict]
):
    """
    Background task to execute model training.
    
    This function is called asynchronously after the training run is created.
    It uses the TrainingService to orchestrate the actual training.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        # Import training service
        from app.services.ml.training_service import get_training_service
        from app.core.database import db_manager
        from app.models import TrainingRun, TaskStatus
        from sqlalchemy import select
        
        logger.info(f"Starting training: {sport_code} {bet_type} {framework}")
        
        # Get training service
        service = get_training_service()
        
        # Run training
        result = await service.train_model(
            sport_code=sport_code,
            bet_type=bet_type,
            framework=framework,
            max_runtime_secs=max_runtime,
            save_to_db=True,  # Will create/update model records
        )
        
        # Update the original training run with results
        await db_manager.initialize()
        async with db_manager.session() as session:
            query = select(TrainingRun).where(TrainingRun.id == training_run_id)
            db_result = await session.execute(query)
            training_run = db_result.scalar_one_or_none()
            
            if training_run:
                if result.success:
                    training_run.status = TaskStatus.SUCCESS
                    training_run.validation_metrics = {
                        "auc": result.auc,
                        "accuracy": result.accuracy,
                        "log_loss": result.log_loss,
                        "training_samples": result.training_samples,
                        "feature_count": result.feature_count,
                    }
                else:
                    training_run.status = TaskStatus.FAILED
                    training_run.error_message = result.error_message
                
                training_run.completed_at = datetime.utcnow()
                training_run.training_duration_seconds = int(result.training_duration_seconds)
                await session.commit()
        
        logger.info(f"Training complete: {sport_code} {bet_type} - Success: {result.success}")
        
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        
        # Update training run as failed
        try:
            from app.core.database import db_manager
            from app.models import TrainingRun, TaskStatus
            from sqlalchemy import select
            
            await db_manager.initialize()
            async with db_manager.session() as session:
                query = select(TrainingRun).where(TrainingRun.id == training_run_id)
                db_result = await session.execute(query)
                training_run = db_result.scalar_one_or_none()
                
                if training_run:
                    training_run.status = TaskStatus.FAILED
                    training_run.error_message = str(e)
                    training_run.completed_at = datetime.utcnow()
                    await session.commit()
        except:
            pass


@router.get("/training/{run_id}", response_model=TrainingRunResponse)
async def get_training_status(
    run_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get status of a training run.
    """
    result = await db.execute(
        select(TrainingRun).where(TrainingRun.id == run_id)
    )
    training_run = result.scalar_one_or_none()
    
    if not training_run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Training run not found"
        )
    
    return training_run


@router.get("/training", response_model=List[TrainingRunResponse])
async def list_training_runs(
    sport_code: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 20,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    List recent training runs.
    """
    query = select(TrainingRun)
    
    if sport_code:
        query = query.where(TrainingRun.sport_code == sport_code.upper())
    if status:
        query = query.where(TrainingRun.status == status)
    
    query = query.order_by(desc(TrainingRun.started_at)).limit(limit)
    
    result = await db.execute(query)
    return result.scalars().all()


@router.post("/{model_id}/promote")
async def promote_model(
    model_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles(["admin", "system"]))
):
    """
    Promote a model to production.
    
    **Admin only**
    
    Demotes any existing production model for the same sport/bet_type.
    """
    result = await db.execute(
        select(MLModel).where(MLModel.id == model_id)
    )
    model = result.scalar_one_or_none()
    
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found"
        )
    
    if model.status not in ["ready", "production"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot promote model with status: {model.status}"
        )
    
    # Demote current production model
    await db.execute(
        select(MLModel).where(
            MLModel.sport_code == model.sport_code,
            MLModel.bet_type == model.bet_type,
            MLModel.status == "production",
            MLModel.id != model_id
        )
    )
    result = await db.execute(
        select(MLModel).where(
            MLModel.sport_code == model.sport_code,
            MLModel.bet_type == model.bet_type,
            MLModel.status == "production",
            MLModel.id != model_id
        )
    )
    current_prod = result.scalar_one_or_none()
    if current_prod:
        current_prod.status = "deprecated"
    
    # Promote new model
    model.status = "production"
    model.promoted_at = datetime.utcnow()
    
    await db.commit()
    
    # Clear cache
    await cache.delete("models:production")
    
    return {
        "message": f"Model {model_id} promoted to production",
        "sport_code": model.sport_code,
        "bet_type": model.bet_type
    }


@router.post("/{model_id}/deprecate")
async def deprecate_model(
    model_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles(["admin", "system"]))
):
    """
    Deprecate a model.
    
    **Admin only**
    """
    result = await db.execute(
        select(MLModel).where(MLModel.id == model_id)
    )
    model = result.scalar_one_or_none()
    
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found"
        )
    
    if model.status == "production":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot deprecate production model. Promote another model first."
        )
    
    model.status = "deprecated"
    await db.commit()
    
    return {"message": f"Model {model_id} deprecated"}


@router.get("/compare/{sport_code}/{bet_type}", response_model=ModelComparisonResponse)
async def compare_models(
    sport_code: str,
    bet_type: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Compare all models for a sport/bet_type combination.
    
    Returns performance comparison and recommendation.
    """
    result = await db.execute(
        select(MLModel).where(
            MLModel.sport_code == sport_code.upper(),
            MLModel.bet_type == bet_type.lower(),
            MLModel.status.in_(["ready", "production"])
        )
        .order_by(desc(MLModel.auc))
    )
    models = result.scalars().all()
    
    if not models:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No models found for comparison"
        )
    
    # Calculate comparison metrics
    comparison = {
        "best_auc": max(m.auc or 0 for m in models),
        "best_accuracy": max(m.accuracy or 0 for m in models),
        "frameworks_compared": list(set(m.framework for m in models)),
        "total_models": len(models)
    }
    
    # Find best model
    best_model = max(models, key=lambda m: (m.auc or 0, m.accuracy or 0))
    recommendation = f"Recommend model {best_model.id} ({best_model.framework}) with AUC={best_model.auc:.4f}"
    
    return ModelComparisonResponse(
        models=[ModelBase.model_validate(m) for m in models],
        comparison_metrics=comparison,
        recommendation=recommendation
    )


@router.delete("/{model_id}")
async def delete_model(
    model_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles(["admin"]))
):
    """
    Delete a model.
    
    **Admin only**
    
    Cannot delete production models.
    """
    result = await db.execute(
        select(MLModel).where(MLModel.id == model_id)
    )
    model = result.scalar_one_or_none()
    
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found"
        )
    
    if model.status == "production":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete production model"
        )
    
    await db.delete(model)
    await db.commit()
    
    return {"message": f"Model {model_id} deleted"}
