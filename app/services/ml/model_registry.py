"""
Model Registry Module for LOYALEY

Provides versioning, artifact storage, and performance tracking
for trained ML models.
"""

import os
import json
import shutil
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from pathlib import Path
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Model lifecycle status."""
    TRAINING = 'training'
    VALIDATION = 'validation'
    STAGING = 'staging'
    PRODUCTION = 'production'
    ARCHIVED = 'archived'
    FAILED = 'failed'


class ModelFramework(Enum):
    """Supported ML frameworks."""
    H2O = 'h2o'
    AUTOGLUON = 'autogluon'
    SKLEARN = 'sklearn'
    ENSEMBLE = 'ensemble'


@dataclass
class ModelMetrics:
    """Performance metrics for a model."""
    accuracy: float
    auc: float
    log_loss: float
    brier_score: float
    f1_score: float
    precision: float
    recall: float
    
    # Tier-specific metrics
    tier_a_accuracy: Optional[float] = None
    tier_b_accuracy: Optional[float] = None
    tier_c_accuracy: Optional[float] = None
    
    # Additional metrics
    calibration_error: Optional[float] = None
    clv: Optional[float] = None
    roi: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'accuracy': self.accuracy,
            'auc': self.auc,
            'log_loss': self.log_loss,
            'brier_score': self.brier_score,
            'f1_score': self.f1_score,
            'precision': self.precision,
            'recall': self.recall,
            'tier_a_accuracy': self.tier_a_accuracy,
            'tier_b_accuracy': self.tier_b_accuracy,
            'tier_c_accuracy': self.tier_c_accuracy,
            'calibration_error': self.calibration_error,
            'clv': self.clv,
            'roi': self.roi
        }
        
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ModelMetrics':
        return cls(
            accuracy=d.get('accuracy', 0),
            auc=d.get('auc', 0),
            log_loss=d.get('log_loss', 1),
            brier_score=d.get('brier_score', 0.25),
            f1_score=d.get('f1_score', 0),
            precision=d.get('precision', 0),
            recall=d.get('recall', 0),
            tier_a_accuracy=d.get('tier_a_accuracy'),
            tier_b_accuracy=d.get('tier_b_accuracy'),
            tier_c_accuracy=d.get('tier_c_accuracy'),
            calibration_error=d.get('calibration_error'),
            clv=d.get('clv'),
            roi=d.get('roi')
        )


@dataclass
class TrainingConfig:
    """Configuration used for model training."""
    sport: str
    bet_type: str  # spread, moneyline, total
    features: List[str]
    hyperparameters: Dict[str, Any]
    training_window_days: int
    validation_window_days: int
    min_training_samples: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'sport': self.sport,
            'bet_type': self.bet_type,
            'features': self.features,
            'hyperparameters': self.hyperparameters,
            'training_window_days': self.training_window_days,
            'validation_window_days': self.validation_window_days,
            'min_training_samples': self.min_training_samples
        }
        
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'TrainingConfig':
        return cls(
            sport=d['sport'],
            bet_type=d['bet_type'],
            features=d.get('features', []),
            hyperparameters=d.get('hyperparameters', {}),
            training_window_days=d.get('training_window_days', 365),
            validation_window_days=d.get('validation_window_days', 30),
            min_training_samples=d.get('min_training_samples', 500)
        )


@dataclass
class ModelVersion:
    """Represents a specific version of a model."""
    model_id: str
    version: str
    sport: str
    bet_type: str
    framework: ModelFramework
    status: ModelStatus
    metrics: ModelMetrics
    training_config: TrainingConfig
    artifact_path: str
    artifact_hash: str
    feature_count: int
    training_samples: int
    training_date_start: datetime
    training_date_end: datetime
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    promoted_at: Optional[datetime] = None
    description: str = ''
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_id': self.model_id,
            'version': self.version,
            'sport': self.sport,
            'bet_type': self.bet_type,
            'framework': self.framework.value,
            'status': self.status.value,
            'metrics': self.metrics.to_dict(),
            'training_config': self.training_config.to_dict(),
            'artifact_path': self.artifact_path,
            'artifact_hash': self.artifact_hash,
            'feature_count': self.feature_count,
            'training_samples': self.training_samples,
            'training_date_start': self.training_date_start.isoformat(),
            'training_date_end': self.training_date_end.isoformat(),
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'promoted_at': self.promoted_at.isoformat() if self.promoted_at else None,
            'description': self.description,
            'tags': self.tags
        }
        
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ModelVersion':
        return cls(
            model_id=d['model_id'],
            version=d['version'],
            sport=d['sport'],
            bet_type=d['bet_type'],
            framework=ModelFramework(d['framework']),
            status=ModelStatus(d['status']),
            metrics=ModelMetrics.from_dict(d['metrics']),
            training_config=TrainingConfig.from_dict(d['training_config']),
            artifact_path=d['artifact_path'],
            artifact_hash=d['artifact_hash'],
            feature_count=d['feature_count'],
            training_samples=d['training_samples'],
            training_date_start=datetime.fromisoformat(d['training_date_start']),
            training_date_end=datetime.fromisoformat(d['training_date_end']),
            created_at=datetime.fromisoformat(d['created_at']),
            updated_at=datetime.fromisoformat(d['updated_at']),
            promoted_at=datetime.fromisoformat(d['promoted_at']) if d.get('promoted_at') else None,
            description=d.get('description', ''),
            tags=d.get('tags', [])
        )


@dataclass
class PerformanceRecord:
    """Historical performance record for tracking model accuracy over time."""
    model_id: str
    date: datetime
    predictions_count: int
    accuracy: float
    tier_a_accuracy: Optional[float] = None
    tier_a_count: int = 0
    clv: Optional[float] = None
    roi: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_id': self.model_id,
            'date': self.date.isoformat(),
            'predictions_count': self.predictions_count,
            'accuracy': self.accuracy,
            'tier_a_accuracy': self.tier_a_accuracy,
            'tier_a_count': self.tier_a_count,
            'clv': self.clv,
            'roi': self.roi
        }


class ModelRegistry:
    """
    Central registry for ML model management.
    
    Provides:
    - Model versioning
    - Artifact storage
    - Performance tracking
    - Model promotion workflow
    """
    
    def __init__(self, base_path: str = 'models'):
        """
        Initialize model registry.
        
        Args:
            base_path: Base directory for model storage
        """
        self.base_path = Path(base_path)
        self.models_path = self.base_path / 'artifacts'
        self.registry_path = self.base_path / 'registry'
        self.performance_path = self.base_path / 'performance'
        
        # Create directories
        self.models_path.mkdir(parents=True, exist_ok=True)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.performance_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache
        self._registry_cache: Dict[str, ModelVersion] = {}
        self._load_registry()
        
    def _load_registry(self):
        """Load registry from disk."""
        registry_file = self.registry_path / 'models.json'
        if registry_file.exists():
            try:
                with open(registry_file, 'r') as f:
                    data = json.load(f)
                for model_data in data.get('models', []):
                    model = ModelVersion.from_dict(model_data)
                    self._registry_cache[model.model_id] = model
                logger.info(f"Loaded {len(self._registry_cache)} models from registry")
            except Exception as e:
                logger.error(f"Failed to load registry: {e}")
                
    def _save_registry(self):
        """Save registry to disk."""
        registry_file = self.registry_path / 'models.json'
        data = {
            'models': [m.to_dict() for m in self._registry_cache.values()],
            'updated_at': datetime.utcnow().isoformat()
        }
        with open(registry_file, 'w') as f:
            json.dump(data, f, indent=2)
            
    def _generate_model_id(self, sport: str, bet_type: str, framework: str) -> str:
        """Generate unique model ID."""
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        return f"{sport}_{bet_type}_{framework}_{timestamp}"
        
    def _generate_version(self, sport: str, bet_type: str) -> str:
        """Generate next version number for sport/bet_type combination."""
        existing = self.list_models(sport=sport, bet_type=bet_type)
        if not existing:
            return '1.0.0'
            
        # Find highest version
        versions = []
        for m in existing:
            parts = m.version.split('.')
            if len(parts) == 3:
                try:
                    versions.append((int(parts[0]), int(parts[1]), int(parts[2])))
                except ValueError:
                    continue
                    
        if not versions:
            return '1.0.0'
            
        major, minor, patch = max(versions)
        return f"{major}.{minor}.{patch + 1}"
        
    def _calculate_file_hash(self, path: str) -> str:
        """Calculate SHA-256 hash of file or directory."""
        sha256 = hashlib.sha256()
        
        path_obj = Path(path)
        if path_obj.is_file():
            with open(path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    sha256.update(chunk)
        elif path_obj.is_dir():
            for file_path in sorted(path_obj.rglob('*')):
                if file_path.is_file():
                    with open(file_path, 'rb') as f:
                        for chunk in iter(lambda: f.read(8192), b''):
                            sha256.update(chunk)
                            
        return sha256.hexdigest()
        
    def register_model(
        self,
        artifact_path: str,
        sport: str,
        bet_type: str,
        framework: ModelFramework,
        metrics: ModelMetrics,
        training_config: TrainingConfig,
        feature_count: int,
        training_samples: int,
        training_date_start: datetime,
        training_date_end: datetime,
        description: str = '',
        tags: List[str] = None
    ) -> ModelVersion:
        """
        Register a new model version.
        
        Args:
            artifact_path: Path to model artifact
            sport: Sport code
            bet_type: Type of bet
            framework: ML framework used
            metrics: Training/validation metrics
            training_config: Training configuration
            feature_count: Number of features used
            training_samples: Number of training samples
            training_date_start: Start of training data
            training_date_end: End of training data
            description: Model description
            tags: Tags for categorization
            
        Returns:
            Registered ModelVersion
        """
        # Generate IDs
        model_id = self._generate_model_id(sport, bet_type, framework.value)
        version = self._generate_version(sport, bet_type)
        
        # Copy artifact to registry
        artifact_dest = self.models_path / model_id
        source_path = Path(artifact_path)
        
        if source_path.is_file():
            artifact_dest.mkdir(parents=True, exist_ok=True)
            dest_file = artifact_dest / source_path.name
            shutil.copy2(artifact_path, dest_file)
            final_path = str(dest_file)
        elif source_path.is_dir():
            if artifact_dest.exists():
                shutil.rmtree(artifact_dest)
            shutil.copytree(artifact_path, artifact_dest)
            final_path = str(artifact_dest)
        else:
            raise ValueError(f"Artifact path does not exist: {artifact_path}")
            
        # Calculate hash
        artifact_hash = self._calculate_file_hash(final_path)
        
        # Create model version
        model = ModelVersion(
            model_id=model_id,
            version=version,
            sport=sport,
            bet_type=bet_type,
            framework=framework,
            status=ModelStatus.VALIDATION,
            metrics=metrics,
            training_config=training_config,
            artifact_path=final_path,
            artifact_hash=artifact_hash,
            feature_count=feature_count,
            training_samples=training_samples,
            training_date_start=training_date_start,
            training_date_end=training_date_end,
            description=description,
            tags=tags or []
        )
        
        # Add to registry
        self._registry_cache[model_id] = model
        self._save_registry()
        
        logger.info(f"Registered model: {model_id} (version {version})")
        return model
        
    def get_model(self, model_id: str) -> Optional[ModelVersion]:
        """Get model by ID."""
        return self._registry_cache.get(model_id)
        
    def get_production_model(
        self,
        sport: str,
        bet_type: str,
        framework: Optional[ModelFramework] = None
    ) -> Optional[ModelVersion]:
        """Get current production model for sport/bet_type."""
        candidates = self.list_models(
            sport=sport,
            bet_type=bet_type,
            status=ModelStatus.PRODUCTION,
            framework=framework
        )
        
        if not candidates:
            return None
            
        # Return most recently promoted
        return max(candidates, key=lambda m: m.promoted_at or m.created_at)
        
    def list_models(
        self,
        sport: Optional[str] = None,
        bet_type: Optional[str] = None,
        status: Optional[ModelStatus] = None,
        framework: Optional[ModelFramework] = None,
        tags: Optional[List[str]] = None
    ) -> List[ModelVersion]:
        """
        List models with optional filters.
        
        Args:
            sport: Filter by sport
            bet_type: Filter by bet type
            status: Filter by status
            framework: Filter by framework
            tags: Filter by tags (must have all)
            
        Returns:
            List of matching ModelVersions
        """
        results = []
        
        for model in self._registry_cache.values():
            if sport and model.sport != sport:
                continue
            if bet_type and model.bet_type != bet_type:
                continue
            if status and model.status != status:
                continue
            if framework and model.framework != framework:
                continue
            if tags and not all(t in model.tags for t in tags):
                continue
                
            results.append(model)
            
        return sorted(results, key=lambda m: m.created_at, reverse=True)
        
    def promote_to_staging(self, model_id: str) -> ModelVersion:
        """Promote model to staging status."""
        model = self.get_model(model_id)
        if not model:
            raise ValueError(f"Model not found: {model_id}")
            
        if model.status != ModelStatus.VALIDATION:
            logger.warning(f"Model {model_id} is not in validation status")
            
        model.status = ModelStatus.STAGING
        model.updated_at = datetime.utcnow()
        self._save_registry()
        
        logger.info(f"Model {model_id} promoted to staging")
        return model
        
    def promote_to_production(self, model_id: str) -> ModelVersion:
        """
        Promote model to production status.
        
        This will demote any existing production model for the same
        sport/bet_type combination.
        """
        model = self.get_model(model_id)
        if not model:
            raise ValueError(f"Model not found: {model_id}")
            
        # Demote existing production model
        existing = self.get_production_model(model.sport, model.bet_type, model.framework)
        if existing and existing.model_id != model_id:
            existing.status = ModelStatus.ARCHIVED
            existing.updated_at = datetime.utcnow()
            logger.info(f"Archived previous production model: {existing.model_id}")
            
        # Promote new model
        model.status = ModelStatus.PRODUCTION
        model.promoted_at = datetime.utcnow()
        model.updated_at = datetime.utcnow()
        self._save_registry()
        
        logger.info(f"Model {model_id} promoted to production")
        return model
        
    def archive_model(self, model_id: str) -> ModelVersion:
        """Archive a model."""
        model = self.get_model(model_id)
        if not model:
            raise ValueError(f"Model not found: {model_id}")
            
        model.status = ModelStatus.ARCHIVED
        model.updated_at = datetime.utcnow()
        self._save_registry()
        
        logger.info(f"Model {model_id} archived")
        return model
        
    def delete_model(self, model_id: str, delete_artifacts: bool = False):
        """
        Delete a model from registry.
        
        Args:
            model_id: Model ID to delete
            delete_artifacts: Also delete artifact files
        """
        model = self.get_model(model_id)
        if not model:
            raise ValueError(f"Model not found: {model_id}")
            
        if model.status == ModelStatus.PRODUCTION:
            raise ValueError("Cannot delete production model. Archive it first.")
            
        # Delete artifacts if requested
        if delete_artifacts:
            artifact_path = Path(model.artifact_path)
            if artifact_path.exists():
                if artifact_path.is_file():
                    artifact_path.unlink()
                else:
                    shutil.rmtree(artifact_path)
                logger.info(f"Deleted artifacts for {model_id}")
                
        # Remove from registry
        del self._registry_cache[model_id]
        self._save_registry()
        
        logger.info(f"Model {model_id} deleted from registry")
        
    def record_performance(
        self,
        model_id: str,
        date: datetime,
        predictions_count: int,
        accuracy: float,
        tier_a_accuracy: Optional[float] = None,
        tier_a_count: int = 0,
        clv: Optional[float] = None,
        roi: Optional[float] = None
    ):
        """
        Record daily performance for a model.
        
        Args:
            model_id: Model ID
            date: Date of performance
            predictions_count: Number of predictions
            accuracy: Overall accuracy
            tier_a_accuracy: Tier A accuracy
            tier_a_count: Number of Tier A predictions
            clv: Closing line value
            roi: Return on investment
        """
        record = PerformanceRecord(
            model_id=model_id,
            date=date,
            predictions_count=predictions_count,
            accuracy=accuracy,
            tier_a_accuracy=tier_a_accuracy,
            tier_a_count=tier_a_count,
            clv=clv,
            roi=roi
        )
        
        # Save to performance file
        perf_file = self.performance_path / f"{model_id}_performance.jsonl"
        with open(perf_file, 'a') as f:
            f.write(json.dumps(record.to_dict()) + '\n')
            
        logger.debug(f"Recorded performance for {model_id} on {date.date()}")
        
    def get_performance_history(
        self,
        model_id: str,
        days: int = 30
    ) -> List[PerformanceRecord]:
        """
        Get performance history for a model.
        
        Args:
            model_id: Model ID
            days: Number of days to retrieve
            
        Returns:
            List of PerformanceRecords
        """
        perf_file = self.performance_path / f"{model_id}_performance.jsonl"
        if not perf_file.exists():
            return []
            
        records = []
        cutoff = datetime.utcnow().replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        cutoff = cutoff.replace(day=cutoff.day - days) if days > 0 else datetime.min
        
        with open(perf_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    record_date = datetime.fromisoformat(data['date'])
                    if record_date >= cutoff:
                        records.append(PerformanceRecord(
                            model_id=data['model_id'],
                            date=record_date,
                            predictions_count=data['predictions_count'],
                            accuracy=data['accuracy'],
                            tier_a_accuracy=data.get('tier_a_accuracy'),
                            tier_a_count=data.get('tier_a_count', 0),
                            clv=data.get('clv'),
                            roi=data.get('roi')
                        ))
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Invalid performance record: {e}")
                    continue
                    
        return sorted(records, key=lambda r: r.date)
        
    def compare_models(
        self,
        model_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Compare multiple models.
        
        Args:
            model_ids: List of model IDs to compare
            
        Returns:
            Comparison dictionary with metrics
        """
        models = []
        for mid in model_ids:
            model = self.get_model(mid)
            if model:
                models.append(model)
                
        if len(models) < 2:
            raise ValueError("Need at least 2 models to compare")
            
        comparison = {
            'models': [m.to_dict() for m in models],
            'metrics_comparison': {},
            'best_model': None
        }
        
        # Compare key metrics
        metrics = ['accuracy', 'auc', 'log_loss', 'brier_score']
        for metric in metrics:
            values = {}
            for m in models:
                values[m.model_id] = getattr(m.metrics, metric)
            comparison['metrics_comparison'][metric] = values
            
        # Determine best model (by accuracy)
        best = max(models, key=lambda m: m.metrics.accuracy)
        comparison['best_model'] = best.model_id
        
        return comparison
        
    def get_model_lineage(self, model_id: str) -> Dict[str, Any]:
        """
        Get model lineage information.
        
        Args:
            model_id: Model ID
            
        Returns:
            Lineage dictionary with training details
        """
        model = self.get_model(model_id)
        if not model:
            raise ValueError(f"Model not found: {model_id}")
            
        return {
            'model_id': model.model_id,
            'version': model.version,
            'framework': model.framework.value,
            'sport': model.sport,
            'bet_type': model.bet_type,
            'training_config': model.training_config.to_dict(),
            'training_period': {
                'start': model.training_date_start.isoformat(),
                'end': model.training_date_end.isoformat()
            },
            'training_samples': model.training_samples,
            'feature_count': model.feature_count,
            'artifact_hash': model.artifact_hash,
            'created_at': model.created_at.isoformat(),
            'status_history': [
                {'status': model.status.value, 'at': model.updated_at.isoformat()}
            ]
        }


# Example usage
if __name__ == '__main__':
    # Create registry
    registry = ModelRegistry(base_path='./test_models')
    
    # Create sample metrics
    metrics = ModelMetrics(
        accuracy=0.62,
        auc=0.67,
        log_loss=0.65,
        brier_score=0.22,
        f1_score=0.60,
        precision=0.63,
        recall=0.58,
        tier_a_accuracy=0.68
    )
    
    # Create training config
    config = TrainingConfig(
        sport='NBA',
        bet_type='spread',
        features=['elo_diff', 'rest_days', 'h2h_win_pct'],
        hyperparameters={'max_models': 50},
        training_window_days=365,
        validation_window_days=30,
        min_training_samples=500
    )
    
    # Register model (would need actual artifact)
    # model = registry.register_model(
    #     artifact_path='./model.pkl',
    #     sport='NBA',
    #     bet_type='spread',
    #     framework=ModelFramework.H2O,
    #     metrics=metrics,
    #     training_config=config,
    #     feature_count=75,
    #     training_samples=5000,
    #     training_date_start=datetime(2023, 1, 1),
    #     training_date_end=datetime(2024, 1, 1)
    # )
    
    print("Model Registry initialized successfully")
    print(f"Registry path: {registry.base_path}")
