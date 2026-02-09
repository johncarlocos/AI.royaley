"""
ROYALEY - Deep Learning Trainer
TensorFlow and LSTM Models for Time-Series Sports Prediction

Implements:
- LSTM (Long Short-Term Memory) for sequential game patterns
- Dense Neural Networks for feature aggregation
- Hybrid models combining LSTM with traditional features
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

from .config import MLConfig, default_ml_config

logger = logging.getLogger(__name__)

# Check TensorFlow availability
TENSORFLOW_AVAILABLE = False
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    TENSORFLOW_AVAILABLE = True
    # Suppress TF warnings
    tf.get_logger().setLevel('ERROR')
    
    # GPU configuration: allow memory growth to avoid OOM
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"TensorFlow GPU configured: {len(gpus)} GPU(s) available with memory growth")
        except RuntimeError as e:
            logger.warning(f"GPU config error (may already be set): {e}")
    else:
        logger.info("TensorFlow running on CPU (no GPU detected)")
except ImportError:
    logger.warning(
        "TensorFlow not installed. Install with: pip install tensorflow>=2.15.0\n"
        "Deep learning training will not be available until installed."
    )


@dataclass
class DeepLearningModelResult:
    """Result from Deep Learning training"""
    model_id: str
    framework: str = "tensorflow"
    sport_code: str = ""
    bet_type: str = ""
    model_type: str = ""  # 'lstm', 'dense', 'hybrid'
    
    # Architecture info
    architecture: Dict = field(default_factory=dict)
    total_params: int = 0
    trainable_params: int = 0
    
    # Performance metrics
    auc: float = 0.0
    accuracy: float = 0.0
    loss: float = 0.0
    val_auc: float = 0.0
    val_accuracy: float = 0.0
    val_loss: float = 0.0
    
    # Training info
    training_time_secs: float = 0.0
    n_training_samples: int = 0
    n_features: int = 0
    n_epochs_trained: int = 0
    best_epoch: int = 0
    
    # Artifact paths
    model_path: str = ""
    weights_path: str = ""
    history_path: str = ""
    
    # Training history
    training_history: Dict = field(default_factory=dict)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)


class LSTMModel:
    """
    LSTM Model for Time-Series Sports Prediction.
    
    Uses sequences of historical game data to capture
    temporal patterns in team performance.
    """
    
    def __init__(
        self,
        sequence_length: int = 10,
        n_features: int = 60,
        lstm_units: List[int] = None,
        dense_units: List[int] = None,
        dropout_rate: float = 0.3,
        learning_rate: float = 0.001,
    ):
        """
        Initialize LSTM model architecture.
        
        Args:
            sequence_length: Number of historical games in sequence
            n_features: Number of features per game
            lstm_units: List of LSTM layer sizes
            dense_units: List of dense layer sizes
            dropout_rate: Dropout rate for regularization
            learning_rate: Adam optimizer learning rate
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.lstm_units = lstm_units or [128, 64]
        self.dense_units = dense_units or [64, 32]
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
        self._is_built = False
        
    def build(self) -> 'LSTMModel':
        """Build the LSTM model architecture."""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required but not installed")
        
        # Input layer for sequences
        sequence_input = layers.Input(
            shape=(self.sequence_length, self.n_features),
            name='sequence_input'
        )
        
        x = sequence_input
        
        # LSTM layers
        for i, units in enumerate(self.lstm_units):
            return_sequences = (i < len(self.lstm_units) - 1)
            x = layers.LSTM(
                units,
                return_sequences=return_sequences,
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate / 2,
                name=f'lstm_{i}'
            )(x)
            x = layers.BatchNormalization(name=f'bn_lstm_{i}')(x)
        
        # Dense layers
        for i, units in enumerate(self.dense_units):
            x = layers.Dense(units, activation='relu', name=f'dense_{i}')(x)
            x = layers.Dropout(self.dropout_rate, name=f'dropout_{i}')(x)
            x = layers.BatchNormalization(name=f'bn_dense_{i}')(x)
        
        # Output layer
        output = layers.Dense(1, activation='sigmoid', name='output')(x)
        
        # Build model
        self.model = Model(inputs=sequence_input, outputs=output, name='lstm_predictor')
        
        # Compile
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.AUC(name='auc'),
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
            ]
        )
        
        self._is_built = True
        logger.info(f"Built LSTM model with {self.model.count_params()} parameters")
        
        return self
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        epochs: int = 100,
        batch_size: int = 32,
        patience: int = 10,
    ) -> Dict[str, List[float]]:
        """
        Train the LSTM model.
        
        Args:
            X_train: Training sequences (N, seq_len, features)
            y_train: Training labels (N,)
            X_val: Validation sequences
            y_val: Validation labels
            epochs: Maximum epochs
            batch_size: Batch size
            patience: Early stopping patience
            
        Returns:
            Training history dictionary
        """
        if not self._is_built:
            self.build()
        
        callbacks = [
            EarlyStopping(
                monitor='val_auc' if X_val is not None else 'auc',
                patience=patience,
                restore_best_weights=True,
                mode='max'
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=patience // 2,
                min_lr=1e-6
            ),
        ]
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not built/loaded")
        return self.model.predict(X, verbose=0).flatten()
    
    def save(self, path: str) -> str:
        """Save model to disk."""
        if self.model is None:
            raise ValueError("No model to save")
        self.model.save(path)
        return path
    
    @classmethod
    def load(cls, path: str) -> 'LSTMModel':
        """Load model from disk."""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required")
        instance = cls()
        instance.model = keras.models.load_model(path)
        instance._is_built = True
        return instance


class HybridLSTMModel:
    """
    Hybrid Model combining LSTM for sequences with Dense network for static features.
    
    Architecture:
    - LSTM branch: Processes game sequences
    - Dense branch: Processes static features (team stats, ELO, etc.)
    - Concatenation and final classification
    """
    
    def __init__(
        self,
        sequence_length: int = 10,
        n_sequence_features: int = 30,
        n_static_features: int = 40,
        lstm_units: List[int] = None,
        static_dense_units: List[int] = None,
        combined_dense_units: List[int] = None,
        dropout_rate: float = 0.3,
        learning_rate: float = 0.001,
    ):
        """
        Initialize Hybrid model.
        
        Args:
            sequence_length: Number of historical games
            n_sequence_features: Features per game in sequence
            n_static_features: Number of static features
            lstm_units: LSTM layer sizes
            static_dense_units: Dense layer sizes for static features
            combined_dense_units: Dense layer sizes after concatenation
            dropout_rate: Dropout rate
            learning_rate: Learning rate
        """
        self.sequence_length = sequence_length
        self.n_sequence_features = n_sequence_features
        self.n_static_features = n_static_features
        self.lstm_units = lstm_units or [64, 32]
        self.static_dense_units = static_dense_units or [64, 32]
        self.combined_dense_units = combined_dense_units or [64, 32]
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
        self._is_built = False
    
    def build(self) -> 'HybridLSTMModel':
        """Build the hybrid model architecture."""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required")
        
        # Sequence input branch (LSTM)
        sequence_input = layers.Input(
            shape=(self.sequence_length, self.n_sequence_features),
            name='sequence_input'
        )
        
        x_seq = sequence_input
        for i, units in enumerate(self.lstm_units):
            return_sequences = (i < len(self.lstm_units) - 1)
            x_seq = layers.LSTM(
                units,
                return_sequences=return_sequences,
                dropout=self.dropout_rate,
                name=f'lstm_{i}'
            )(x_seq)
            x_seq = layers.BatchNormalization()(x_seq)
        
        # Static features branch (Dense)
        static_input = layers.Input(
            shape=(self.n_static_features,),
            name='static_input'
        )
        
        x_static = static_input
        for i, units in enumerate(self.static_dense_units):
            x_static = layers.Dense(units, activation='relu')(x_static)
            x_static = layers.Dropout(self.dropout_rate)(x_static)
            x_static = layers.BatchNormalization()(x_static)
        
        # Concatenate branches
        combined = layers.concatenate([x_seq, x_static], name='concatenate')
        
        # Combined dense layers
        x = combined
        for i, units in enumerate(self.combined_dense_units):
            x = layers.Dense(units, activation='relu')(x)
            x = layers.Dropout(self.dropout_rate)(x)
            x = layers.BatchNormalization()(x)
        
        # Output
        output = layers.Dense(1, activation='sigmoid', name='output')(x)
        
        # Build model
        self.model = Model(
            inputs=[sequence_input, static_input],
            outputs=output,
            name='hybrid_lstm_predictor'
        )
        
        # Compile
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )
        
        self._is_built = True
        logger.info(f"Built Hybrid LSTM model with {self.model.count_params()} parameters")
        
        return self
    
    def fit(
        self,
        X_seq_train: np.ndarray,
        X_static_train: np.ndarray,
        y_train: np.ndarray,
        X_seq_val: np.ndarray = None,
        X_static_val: np.ndarray = None,
        y_val: np.ndarray = None,
        epochs: int = 100,
        batch_size: int = 32,
        patience: int = 10,
    ) -> Dict[str, List[float]]:
        """Train the hybrid model."""
        if not self._is_built:
            self.build()
        
        callbacks = [
            EarlyStopping(
                monitor='val_auc' if y_val is not None else 'auc',
                patience=patience,
                restore_best_weights=True,
                mode='max'
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if y_val is not None else 'loss',
                factor=0.5,
                patience=patience // 2,
                min_lr=1e-6
            ),
        ]
        
        validation_data = None
        if y_val is not None:
            validation_data = ([X_seq_val, X_static_val], y_val)
        
        history = self.model.fit(
            [X_seq_train, X_static_train], y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history.history
    
    def predict(self, X_seq: np.ndarray, X_static: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not built/loaded")
        return self.model.predict([X_seq, X_static], verbose=0).flatten()
    
    def save(self, path: str) -> str:
        """Save model to disk."""
        self.model.save(path)
        return path
    
    @classmethod
    def load(cls, path: str) -> 'HybridLSTMModel':
        """Load model from disk."""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required")
        instance = cls()
        instance.model = keras.models.load_model(path)
        instance._is_built = True
        return instance


class DenseNeuralNetwork:
    """
    Dense Neural Network for feature-based prediction.
    
    Simple but effective architecture for tabular sports data.
    """
    
    def __init__(
        self,
        n_features: int = 60,
        hidden_units: List[int] = None,
        dropout_rate: float = 0.3,
        learning_rate: float = 0.001,
    ):
        """Initialize Dense Neural Network."""
        self.n_features = n_features
        self.hidden_units = hidden_units or [256, 128, 64, 32]
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
        self._is_built = False
    
    def build(self) -> 'DenseNeuralNetwork':
        """Build the model."""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required")
        
        inputs = layers.Input(shape=(self.n_features,), name='features')
        
        x = inputs
        for i, units in enumerate(self.hidden_units):
            x = layers.Dense(units, activation='relu', name=f'dense_{i}')(x)
            x = layers.BatchNormalization(name=f'bn_{i}')(x)
            x = layers.Dropout(self.dropout_rate, name=f'dropout_{i}')(x)
        
        output = layers.Dense(1, activation='sigmoid', name='output')(x)
        
        self.model = Model(inputs=inputs, outputs=output, name='dense_predictor')
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )
        
        self._is_built = True
        logger.info(f"Built Dense NN with {self.model.count_params()} parameters")
        
        return self
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        epochs: int = 100,
        batch_size: int = 32,
        patience: int = 10,
    ) -> Dict[str, List[float]]:
        """Train the model."""
        if not self._is_built:
            self.build()
        
        callbacks = [
            EarlyStopping(
                monitor='val_auc' if X_val is not None else 'auc',
                patience=patience,
                restore_best_weights=True,
                mode='max'
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=patience // 2,
                min_lr=1e-6
            ),
        ]
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X, verbose=0).flatten()
    
    def save(self, path: str) -> str:
        """Save model."""
        self.model.save(path)
        return path
    
    @classmethod
    def load(cls, path: str) -> 'DenseNeuralNetwork':
        """Load model."""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required")
        instance = cls()
        instance.model = keras.models.load_model(path)
        instance._is_built = True
        return instance


class DeepLearningTrainer:
    """
    Deep Learning trainer for sports prediction.
    
    Supports:
    - LSTM for time-series patterns
    - Dense NN for tabular features
    - Hybrid LSTM+Dense models
    """
    
    def __init__(
        self,
        config: MLConfig = None,
        model_dir: str = None,
    ):
        """Initialize Deep Learning trainer."""
        self.config = config or default_ml_config
        self.model_dir = Path(model_dir or self.config.model_artifact_path) / "tensorflow"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._model = None
        self._model_type = None
        self._scaler = None
        self._feature_columns = None
        self._model_path = None
    
    def train_lstm(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        sport_code: str,
        bet_type: str,
        val_sequences: np.ndarray = None,
        val_labels: np.ndarray = None,
        sequence_length: int = 10,
        lstm_units: List[int] = None,
        epochs: int = 100,
        batch_size: int = 32,
    ) -> DeepLearningModelResult:
        """
        Train LSTM model for time-series prediction.
        
        Args:
            sequences: Training sequences (N, seq_len, features)
            labels: Training labels
            sport_code: Sport code
            bet_type: Bet type
            val_sequences: Validation sequences
            val_labels: Validation labels
            sequence_length: Sequence length
            lstm_units: LSTM layer sizes
            epochs: Max epochs
            batch_size: Batch size
            
        Returns:
            DeepLearningModelResult
        """
        logger.info(f"Training LSTM for {sport_code} {bet_type}")
        start_time = datetime.now(timezone.utc)
        
        n_features = sequences.shape[2]
        
        # Build model
        model = LSTMModel(
            sequence_length=sequence_length,
            n_features=n_features,
            lstm_units=lstm_units or [128, 64],
        )
        model.build()
        
        # Train
        history = model.fit(
            sequences, labels,
            X_val=val_sequences,
            y_val=val_labels,
            epochs=epochs,
            batch_size=batch_size,
        )
        
        self._model = model
        self._model_type = 'lstm'
        
        # Save model
        model_path = self.model_dir / sport_code / bet_type / "lstm_model"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(model_path))
        
        # Get best metrics
        best_epoch = np.argmax(history.get('val_auc', history.get('auc', [0])))
        
        training_time = (datetime.now(timezone.utc) - start_time).total_seconds()
        
        return DeepLearningModelResult(
            model_id=f"lstm_{sport_code}_{bet_type}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
            sport_code=sport_code,
            bet_type=bet_type,
            model_type='lstm',
            architecture={
                'sequence_length': sequence_length,
                'n_features': n_features,
                'lstm_units': model.lstm_units,
                'dense_units': model.dense_units,
            },
            total_params=model.model.count_params(),
            trainable_params=sum([np.prod(w.shape) for w in model.model.trainable_weights]),
            auc=max(history.get('auc', [0])),
            accuracy=max(history.get('accuracy', [0])),
            loss=min(history.get('loss', [1])),
            val_auc=max(history.get('val_auc', [0])) if 'val_auc' in history else 0,
            val_accuracy=max(history.get('val_accuracy', [0])) if 'val_accuracy' in history else 0,
            val_loss=min(history.get('val_loss', [1])) if 'val_loss' in history else 0,
            training_time_secs=training_time,
            n_training_samples=len(labels),
            n_features=n_features,
            n_epochs_trained=len(history.get('loss', [])),
            best_epoch=best_epoch,
            model_path=str(model_path),
            training_history=history,
        )
    
    def train(
        self,
        train_df: pd.DataFrame,
        target_column: str,
        feature_columns: List[str],
        sport_code: str,
        bet_type: str,
        validation_df: pd.DataFrame = None,
        max_epochs: int = 100,
        early_stopping_patience: int = 10,
        fast_mode: bool = False,
        **kwargs,
    ) -> DeepLearningModelResult:
        """
        Train Deep Learning model from DataFrame inputs.
        
        Main entry point called by training_service.
        Handles DataFrame → numpy conversion, NaN imputation, feature scaling,
        and delegates to train_dense().
        """
        from sklearn.preprocessing import StandardScaler
        
        if fast_mode:
            max_epochs = min(max_epochs, 30)
            early_stopping_patience = 5
        
        logger.info(
            f"Deep Learning train() for {sport_code} {bet_type}: "
            f"{len(train_df)} samples, {len(feature_columns)} features, "
            f"epochs={max_epochs}, patience={early_stopping_patience}"
        )
        
        # Log GPU status
        if TENSORFLOW_AVAILABLE:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                logger.info(f"🎮 Training on GPU: {gpus[0].name}")
            else:
                logger.info("🖥️ Training on CPU (no GPU detected)")
        
        # Extract numpy arrays
        X_train = train_df[feature_columns].values.astype(np.float32)
        y_train = train_df[target_column].values.astype(np.float32)
        
        X_val = None
        y_val = None
        if validation_df is not None and len(validation_df) > 0:
            X_val = validation_df[feature_columns].values.astype(np.float32)
            y_val = validation_df[target_column].values.astype(np.float32)
        
        # Handle NaN/Inf
        nan_count_train = np.isnan(X_train).sum()
        if nan_count_train > 0:
            logger.warning(f"Replacing {nan_count_train} NaN values in training features")
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        if X_val is not None:
            X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Feature scaling - critical for neural networks
        self._scaler = StandardScaler()
        X_train = self._scaler.fit_transform(X_train)
        if X_val is not None:
            X_val = self._scaler.transform(X_val)
        
        # Save scaler for prediction
        scaler_path = self.model_dir / sport_code / bet_type / "scaler.pkl"
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        import pickle
        with open(scaler_path, 'wb') as f:
            pickle.dump(self._scaler, f)
        
        # Store for prediction
        self._feature_columns = feature_columns
        self._model_path = str(self.model_dir / sport_code / bet_type / "dense_model")
        
        # Train Dense NN (tabular data → Dense is the right architecture)
        result = self.train_dense(
            X_train=X_train,
            y_train=y_train,
            sport_code=sport_code,
            bet_type=bet_type,
            X_val=X_val,
            y_val=y_val,
            epochs=max_epochs,
            batch_size=min(64, max(16, len(X_train) // 50)),
        )
        
        # Use validation metrics as primary (more realistic)
        if result.val_auc > 0:
            result.auc = result.val_auc
        if result.val_accuracy > 0:
            result.accuracy = result.val_accuracy
        
        logger.info(
            f"Deep Learning complete: AUC={result.auc:.4f}, "
            f"Accuracy={result.accuracy:.4f}, "
            f"Epochs={result.n_epochs_trained}, Time={result.training_time_secs:.1f}s"
        )
        
        return result
    
    def predict_with_loaded(
        self,
        data: pd.DataFrame,
        feature_columns: List[str],
    ) -> np.ndarray:
        """Predict using already-loaded model (for walk-forward validation)."""
        if self._model is None:
            raise ValueError("No model loaded. Call train() first.")
        
        X = data[feature_columns].values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        if hasattr(self, '_scaler') and self._scaler is not None:
            X = self._scaler.transform(X)
        
        return self._model.predict(X)
    
    def train_dense(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        sport_code: str,
        bet_type: str,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        hidden_units: List[int] = None,
        epochs: int = 100,
        batch_size: int = 32,
    ) -> DeepLearningModelResult:
        """Train Dense Neural Network."""
        logger.info(f"Training Dense NN for {sport_code} {bet_type}")
        start_time = datetime.now(timezone.utc)
        
        n_features = X_train.shape[1]
        
        model = DenseNeuralNetwork(
            n_features=n_features,
            hidden_units=hidden_units or [256, 128, 64, 32],
        )
        model.build()
        
        history = model.fit(
            X_train, y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=epochs,
            batch_size=batch_size,
        )
        
        self._model = model
        self._model_type = 'dense'
        
        model_path = self.model_dir / sport_code / bet_type / "dense_model"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(model_path))
        
        best_epoch = np.argmax(history.get('val_auc', history.get('auc', [0])))
        training_time = (datetime.now(timezone.utc) - start_time).total_seconds()
        
        return DeepLearningModelResult(
            model_id=f"dense_{sport_code}_{bet_type}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
            sport_code=sport_code,
            bet_type=bet_type,
            model_type='dense',
            architecture={
                'n_features': n_features,
                'hidden_units': model.hidden_units,
            },
            total_params=model.model.count_params(),
            trainable_params=sum([np.prod(w.shape) for w in model.model.trainable_weights]),
            auc=max(history.get('auc', [0])),
            accuracy=max(history.get('accuracy', [0])),
            loss=min(history.get('loss', [1])),
            val_auc=max(history.get('val_auc', [0])) if 'val_auc' in history else 0,
            val_accuracy=max(history.get('val_accuracy', [0])) if 'val_accuracy' in history else 0,
            val_loss=min(history.get('val_loss', [1])) if 'val_loss' in history else 0,
            training_time_secs=training_time,
            n_training_samples=len(y_train),
            n_features=n_features,
            n_epochs_trained=len(history.get('loss', [])),
            best_epoch=best_epoch,
            model_path=str(model_path),
            training_history=history,
        )
    
    def predict(self, model_path: str = None, data: pd.DataFrame = None, feature_columns: List[str] = None) -> np.ndarray:
        """Make predictions using loaded or specified model.
        
        Supports two calling conventions:
        - predict(X) - numpy array, uses already-loaded model
        - predict(model_path, data, feature_columns) - loads model and predicts
        """
        # Handle legacy call: predict(X_numpy_array)
        if model_path is not None and isinstance(model_path, np.ndarray):
            if self._model is None:
                raise ValueError("No model loaded")
            return self._model.predict(model_path)
        
        # Standard interface: predict(model_path, data, feature_columns)
        if data is not None and feature_columns is not None:
            # Load model if not already loaded
            if self._model is None and model_path:
                model_dir = Path(model_path)
                if model_dir.is_dir():
                    # Check for dense_model subdirectory
                    dense_path = model_dir / "dense_model"
                    if dense_path.exists():
                        self.load(str(dense_path), model_type='dense')
                    else:
                        self.load(str(model_dir), model_type='dense')
                else:
                    self.load(str(model_dir.parent), model_type='dense')
            
            if self._model is None:
                raise ValueError("No model loaded")
            
            X = data[feature_columns].values.astype(np.float32)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            if hasattr(self, '_scaler') and self._scaler is not None:
                X = self._scaler.transform(X)
            
            return self._model.predict(X)
        
        raise ValueError("Invalid arguments: provide either (X) or (model_path, data, feature_columns)")
    
    def load(self, model_path: str, model_type: str = 'lstm') -> None:
        """Load a saved model."""
        if model_type == 'lstm':
            self._model = LSTMModel.load(model_path)
        elif model_type == 'dense':
            self._model = DenseNeuralNetwork.load(model_path)
        elif model_type == 'hybrid':
            self._model = HybridLSTMModel.load(model_path)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        self._model_type = model_type


class DeepLearningTrainerMock:
    """Mock trainer for testing without TensorFlow."""
    
    def __init__(self, config: MLConfig = None, model_dir: str = None):
        self.config = config or default_ml_config
        self.model_dir = Path(model_dir or "./models/tensorflow_mock")
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
        **kwargs,
    ) -> DeepLearningModelResult:
        """Mock train() for testing without TensorFlow."""
        logger.info(f"Mock Deep Learning training for {sport_code} {bet_type}")
        return DeepLearningModelResult(
            model_id=f"mock_dl_{sport_code}_{bet_type}",
            sport_code=sport_code,
            bet_type=bet_type,
            model_type='dense',
            auc=0.56 + np.random.random() * 0.08,
            accuracy=0.53 + np.random.random() * 0.08,
            training_time_secs=60.0,
            n_training_samples=len(train_df),
            n_features=len(feature_columns),
            model_path=str(self.model_dir / sport_code / bet_type / "mock_model"),
        )
    
    def predict_with_loaded(self, data: pd.DataFrame, feature_columns: List[str]) -> np.ndarray:
        """Mock prediction."""
        return np.random.beta(2, 2, len(data))
    
    def train_lstm(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        sport_code: str,
        bet_type: str,
        **kwargs,
    ) -> DeepLearningModelResult:
        """Mock LSTM training."""
        logger.info(f"Mock LSTM training for {sport_code} {bet_type}")
        return DeepLearningModelResult(
            model_id=f"mock_lstm_{sport_code}_{bet_type}",
            sport_code=sport_code,
            bet_type=bet_type,
            model_type='lstm',
            auc=0.58 + np.random.random() * 0.08,
            accuracy=0.55 + np.random.random() * 0.08,
            training_time_secs=180.0,
            n_training_samples=len(labels),
        )
    
    def train_dense(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        sport_code: str,
        bet_type: str,
        **kwargs,
    ) -> DeepLearningModelResult:
        """Mock Dense training."""
        logger.info(f"Mock Dense training for {sport_code} {bet_type}")
        return DeepLearningModelResult(
            model_id=f"mock_dense_{sport_code}_{bet_type}",
            sport_code=sport_code,
            bet_type=bet_type,
            model_type='dense',
            auc=0.57 + np.random.random() * 0.08,
            accuracy=0.54 + np.random.random() * 0.08,
            training_time_secs=120.0,
            n_training_samples=len(y_train),
        )
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Mock prediction."""
        return np.random.beta(2, 2, len(X))


def get_deep_learning_trainer(
    config: MLConfig = None,
    use_mock: bool = False,
) -> Union[DeepLearningTrainer, DeepLearningTrainerMock]:
    """
    Factory function to get Deep Learning trainer.
    
    Args:
        config: ML configuration
        use_mock: Use mock trainer for testing
        
    Returns:
        Deep Learning trainer instance
    """
    if use_mock:
        return DeepLearningTrainerMock(config)
    
    if TENSORFLOW_AVAILABLE:
        return DeepLearningTrainer(config)
    else:
        logger.warning("TensorFlow not installed, using mock trainer")
        return DeepLearningTrainerMock(config)